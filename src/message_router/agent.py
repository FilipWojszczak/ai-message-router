"""The routing agent: one model call picks the department, a tool sends the email.

The model controls only the department (an enum). The requester email, the message
body and the settings reach the tool through the runtime context, never through model
arguments. If the run ends without a send, the code falls back to other@ itself.
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx
import ollama
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware.types import InputAgentState, OutputAgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

from message_router.config import Settings
from message_router.departments import Department
from message_router.mailer import EmailDeliveryError, send_department_email
from message_router.prompts import SYSTEM_PROMPT, format_user_message

# uvicorn configures this logger, so messages show up without extra logging setup.
logger = logging.getLogger("uvicorn.error")

# Fixed, neutral user text for the startup warm-up; only the prefix before it matters.
WARMUP_MESSAGE: Final[str] = "Startup warm-up request."


class LLMUnavailableError(Exception):
    """Ollama could not be reached or dropped the connection (HTTP 503)."""


class LLMTimeoutError(Exception):
    """Ollama stalled, or the whole agent run exceeded its deadline (HTTP 504)."""


class LLMResponseError(Exception):
    """Ollama answered with an error, e.g. the model is not pulled (HTTP 502)."""


@dataclass
class RoutingContext:
    """Per-request data for the tool. Passed by reference, so it is also the result."""

    requester_email: str
    message: str
    settings: Settings
    # Written by the tool; read by route_message after the run.
    department: Department | None = None
    message_id: str | None = None


@dataclass(frozen=True)
class RoutingResult:
    department: Department
    fallback: bool
    message_id: str


type RouterAgent = CompiledStateGraph[
    AgentState[Any], RoutingContext, InputAgentState, OutputAgentState[Any]
]


@tool(return_direct=True)
async def send_email(
    department: Department, runtime: ToolRuntime[RoutingContext]
) -> str:
    """Send the user's message to the chosen department."""
    ctx = runtime.context
    # Check and set before the first await: parallel tool calls from one model
    # response run as concurrent tasks, and only the first may send.
    if ctx.department is not None:
        return "Already sent; do not call again."
    ctx.department = Department(department)
    ctx.message_id = await send_department_email(
        ctx.department, ctx.requester_email, ctx.message, settings=ctx.settings
    )
    return f"Sent to {ctx.department.value}."


def build_chat_model(settings: Settings) -> ChatOllama:
    """Build the Ollama chat model. No I/O (the model is not validated on init)."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_url,
        temperature=0,
        reasoning=settings.ollama_reasoning,
        num_predict=settings.ollama_num_predict,
        client_kwargs={"timeout": settings.ollama_timeout},
    )


def build_agent(model: BaseChatModel) -> RouterAgent:
    """Compile the agent graph once; it is reused by every request."""
    return create_agent(
        model, [send_email], system_prompt=SYSTEM_PROMPT, context_schema=RoutingContext
    )


# The deadline is a parameter, not the caller's asyncio.timeout, because hitting it
# must be logged and swallowed here like any other warm-up failure.
async def warm_up_prompt_cache(model: BaseChatModel, *, timeout: float) -> None:  # noqa: ASYNC109
    """Fill Ollama's prompt prefix cache with the system prompt and tool schema.

    One model call with the same prefix as a real request (create_agent binds the
    tools the same way); the answer is discarded and the tool never runs, since this
    is not an agent run. Best-effort: logs and returns on any error.
    """
    start = time.perf_counter()
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(format_user_message(WARMUP_MESSAGE)),
    ]
    try:
        async with asyncio.timeout(timeout):
            await model.bind_tools([send_email]).ainvoke(messages)
    # Not BaseException: a CancelledError must propagate. (uvicorn does not cancel the
    # lifespan on SIGTERM: it defers the exit until startup ends, so ollama_timeout
    # bounds the wait.)
    except Exception as exc:
        # Only the type: the error text may come from Ollama.
        logger.warning(
            "Prompt cache warm-up failed: error=%s latency_ms=%d",
            type(exc).__name__,
            _elapsed_ms(start),
        )
        return
    logger.info("Prompt cache warmed in %d ms", _elapsed_ms(start))


def _last_ai_message(messages: Sequence[AnyMessage]) -> AIMessage | None:
    return next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)


def _fallback_reason(last_ai: AIMessage | None) -> str:
    if last_ai is not None and (last_ai.tool_calls or last_ai.invalid_tool_calls):
        return "invalid_tool_call"
    return "no_tool_call"


def _elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


async def route_message(
    agent: RouterAgent, requester_email: str, message: str, *, settings: Settings
) -> RoutingResult:
    """Run the agent once, fall back to other@ if it sent nothing.

    Never logs the message or the requester email.

    Raises LLMUnavailableError, LLMTimeoutError, LLMResponseError, EmailDeliveryError.
    """
    start = time.perf_counter()
    ctx = RoutingContext(requester_email, message, settings)
    state = None
    try:
        async with asyncio.timeout(settings.ollama_timeout):
            state = await agent.ainvoke(
                {"messages": [HumanMessage(format_user_message(message))]},
                context=ctx,
            )
    # httpx.TimeoutException is a TransportError subclass: it must come first.
    except (TimeoutError, httpx.TimeoutException) as exc:
        if ctx.message_id is None:
            if ctx.department is not None:
                logger.warning(
                    "Deadline hit while sending to %s: delivery unknown",
                    ctx.department.value,
                )
            _log_failure(exc, 504, start)
            raise LLMTimeoutError("The language model timed out.") from exc
        # The email was sent; only the graph's post-tool steps were cut short.
        # Report success: a 504 here would invite a retry and a duplicate email.
        logger.warning("Deadline hit after the email was sent: reporting success")
    except httpx.TransportError as exc:
        _log_failure(exc, 503, start)
        raise LLMUnavailableError("The language model is unavailable.") from exc
    except ollama.ResponseError as exc:
        # Log only the status code: the error text comes from Ollama.
        _log_failure(exc, 502, start, ollama_status=exc.status_code)
        raise LLMResponseError("The language model returned an error.") from exc
    except EmailDeliveryError as exc:
        # The tool's send failed: no fallback, that would be a second send attempt.
        _log_failure(exc, 502, start, error=str(exc))  # safe text by design
        raise

    last_ai = _last_ai_message(state["messages"]) if state is not None else None
    if ctx.department is not None and ctx.message_id is not None:
        department, message_id, reason = ctx.department, ctx.message_id, None
    else:
        department, reason = Department.OTHER, _fallback_reason(last_ai)
        try:
            message_id = await send_department_email(
                department, requester_email, message, settings=settings
            )
        except EmailDeliveryError as exc:
            _log_failure(exc, 502, start, error=str(exc), fallback_reason=reason)
            raise

    usage = last_ai.usage_metadata if last_ai is not None else None
    logger.info(
        "Routed message: department=%s fallback=%s reason=%s latency_ms=%d model=%s "
        "input_tokens=%s output_tokens=%s",
        department.value,
        reason is not None,
        reason,
        _elapsed_ms(start),
        settings.ollama_model,
        usage["input_tokens"] if usage else None,
        usage["output_tokens"] if usage else None,
    )
    return RoutingResult(
        department=department, fallback=reason is not None, message_id=message_id
    )


def _log_failure(exc: BaseException, status: int, start: float, **details: Any) -> None:
    extra = "".join(f" {key}={value}" for key, value in details.items())
    logger.warning(
        "Routing failed: error=%s status=%d latency_ms=%d%s",
        type(exc).__name__,
        status,
        _elapsed_ms(start),
        extra,
    )
