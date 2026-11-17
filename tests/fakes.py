"""Fakes shared by the agent and API unit tests."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, override

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import tool_call as make_tool_call
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field


@dataclass
class FakeSend:
    """Stands in for ``aiosmtplib.send``: records calls, optionally sleeps or raises."""

    calls: list[tuple[EmailMessage, dict[str, Any]]] = field(default_factory=list)
    error: Exception | None = None
    delay: float = 0.0

    async def __call__(
        self, message: EmailMessage, **kwargs: Any
    ) -> tuple[dict[str, Any], str]:
        self.calls.append((message, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return {}, "OK"


class ToolCallingFakeModel(GenericFakeChatModel):
    """Scripted chat model that accepts tools and records what it was sent.

    Script it with ``messages=iter([AIMessage(...), ...])``; ``error`` and ``delay``
    make the async call raise or stall instead.
    """

    received: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Sequence[Any]] = Field(default_factory=list)
    error: Exception | None = None
    delay: float = 0.0

    @override
    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The base class raises NotImplementedError; the script decides the calls.
        self.bound_tools.append(list(tools))
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self._generate(messages, stop=stop, **kwargs)


def fake_model(*responses: AIMessage | str) -> ToolCallingFakeModel:
    return ToolCallingFakeModel(messages=iter(responses))


def raising_model(
    error: Exception | None = None, delay: float = 0.0
) -> ToolCallingFakeModel:
    return ToolCallingFakeModel(messages=iter([]), error=error, delay=delay)


def tool_call(*departments: str, **extra: Any) -> AIMessage:
    """An AIMessage calling ``send_email`` once per department, with extra args."""
    return AIMessage(
        content="",
        tool_calls=[
            make_tool_call(
                name="send_email",
                args={"department": department, **extra},
                id=f"call_{i}",
            )
            for i, department in enumerate(departments)
        ],
    )
