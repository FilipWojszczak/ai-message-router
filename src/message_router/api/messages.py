from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from message_router.agent import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    RouterAgent,
    route_message,
)
from message_router.config import Settings, get_settings
from message_router.departments import Department
from message_router.mailer import EmailDeliveryError, normalize_requester_email

router = APIRouter(tags=["messages"])

# Code points, as pydantic counts. Keeps the prompt inside a 4096-token context even
# for ~1 token/char text (emoji), so Ollama never truncates away the system prompt.
MAX_MESSAGE_LENGTH: Final = 2000

# Fixed texts: never the exception text (no hostnames, no user data).
LLM_UNAVAILABLE: Final = "The language model is unavailable."
LLM_TIMEOUT: Final = "The language model timed out."
LLM_ERROR: Final = "The language model returned an error."
DELIVERY_FAILED: Final = "Email delivery failed."


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("The message must not be blank.")
    return value


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: Annotated[
        str,
        Field(
            json_schema_extra={"format": "email"}, examples=["jan.nowak@example.com"]
        ),
        AfterValidator(normalize_requester_email),
    ]
    message: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_MESSAGE_LENGTH,
            examples=["Nie działa mi komputer"],
        ),
        AfterValidator(_reject_blank),
    ]


class MessageResponse(BaseModel):
    department: Department
    fallback: bool
    message_id: str


class ErrorResponse(BaseModel):
    detail: str


def get_agent(request: Request) -> RouterAgent:
    """The agent compiled once in the lifespan."""
    return request.app.state.agent


_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    502: {
        "model": ErrorResponse,
        "description": "The language model returned an error, or email delivery "
        "failed (when the model had already chosen a department, the email may "
        "not have been sent; it is never retried).",
    },
    503: {
        "model": ErrorResponse,
        "description": "The language model is unavailable. Nothing was sent.",
    },
    504: {
        "model": ErrorResponse,
        "description": "The language model timed out. Usually nothing was sent.",
    },
}


@router.post("/messages", responses=_ERROR_RESPONSES)
async def route(
    body: MessageRequest,
    agent: Annotated[RouterAgent, Depends(get_agent)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> MessageResponse:
    """Route the message to one department by email, with Reply-To set to `email`.

    The language model picks the department and sends the email through a tool; if it
    doesn't, the message goes to other@example.com (`fallback: true`).
    """
    try:
        result = await route_message(agent, body.email, body.message, settings=settings)
    except LLMUnavailableError as exc:
        raise HTTPException(503, LLM_UNAVAILABLE) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(504, LLM_TIMEOUT) from exc
    except LLMResponseError as exc:
        raise HTTPException(502, LLM_ERROR) from exc
    except EmailDeliveryError as exc:
        raise HTTPException(502, DELIVERY_FAILED) from exc
    return MessageResponse(
        department=result.department,
        fallback=result.fallback,
        message_id=result.message_id,
    )
