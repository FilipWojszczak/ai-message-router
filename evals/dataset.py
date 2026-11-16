"""The labeled routing cases: one JSON object per line in ``cases.jsonl``."""

import json
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from message_router.api.messages import MAX_MESSAGE_LENGTH
from message_router.departments import Department

CASES_PATH: Final = Path(__file__).parent / "cases.jsonl"

# The task's two example messages: must be in the set verbatim, in the dev split.
TASK_EXAMPLES: Final = ("Nie działa mi komputer", "Chciałbym zgłosić urlop na jutro")

# A "long" case exercises the context window, so it must be near the API limit.
LONG_MIN_LENGTH: Final = 1800

type Tag = Literal[
    "task_example", "injection", "ambiguous", "garbled", "off_topic", "long"
]


class DatasetError(ValueError):
    """The cases file is malformed or breaks a dataset rule."""


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)]
    lang: Literal["pl", "en", "mixed"]
    # Accepted set: any of these counts as correct. One item, two only if ambiguous.
    expected: Annotated[tuple[Department, ...], Field(min_length=1, max_length=2)]
    split: Literal["dev", "holdout"]
    tags: frozenset[Tag] = frozenset()
    # The department an injection case tries to force.
    injected: Department | None = None

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> Self:
        if not self.message.strip():
            raise ValueError("message must not be blank")
        if len(set(self.expected)) != len(self.expected):
            raise ValueError("expected must not repeat a department")
        if (len(self.expected) > 1) != ("ambiguous" in self.tags):
            raise ValueError("two expected departments iff tagged 'ambiguous'")
        if (self.injected is not None) != ("injection" in self.tags):
            raise ValueError("injected must be set iff tagged 'injection'")
        if self.injected in self.expected:
            raise ValueError("injected must not be one of the expected departments")
        if "long" in self.tags and len(self.message) < LONG_MIN_LENGTH:
            raise ValueError(f"a 'long' case needs >= {LONG_MIN_LENGTH} characters")
        return self


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    """Read and validate the cases file.

    Raises DatasetError; for a bad line the message names its line number.
    """
    cases: list[Case] = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                cases.append(Case.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise DatasetError(f"{path.name} line {number}: {exc}") from exc

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"{path.name}: duplicate id {case.id!r}")
        seen.add(case.id)

    dev_task_examples = {
        case.message
        for case in cases
        if case.split == "dev" and "task_example" in case.tags
    }
    for example in TASK_EXAMPLES:
        if example not in dev_task_examples:
            raise DatasetError(
                f"{path.name}: task example {example!r} must be present verbatim, "
                "tagged 'task_example', in the dev split"
            )
    return cases
