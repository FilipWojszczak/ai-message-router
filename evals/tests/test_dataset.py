import json
from pathlib import Path
from typing import Any

import pytest

from evals.dataset import TASK_EXAMPLES, DatasetError, load_cases
from message_router.api.messages import MAX_MESSAGE_LENGTH
from message_router.departments import Department

pytestmark = pytest.mark.unit


def _case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": "printer",
        "message": "Drukarka nie drukuje",
        "lang": "pl",
        "expected": ["help-desk@example.com"],
        "split": "dev",
    }
    return case | overrides


def _task_examples() -> list[dict[str, Any]]:
    return [
        _case(id=f"task-example-{i}", message=message, tags=["task_example"])
        for i, message in enumerate(TASK_EXAMPLES)
    ]


def _write(tmp_path: Path, *lines: dict[str, Any] | str) -> Path:
    path = tmp_path / "cases.jsonl"
    text = "\n".join(
        line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)
        for line in lines
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_committed_cases_are_valid() -> None:
    cases = load_cases()

    assert 25 <= len(cases) <= 35
    assert {case.split for case in cases} == {"dev", "holdout"}
    assert {
        case.message
        for case in cases
        if case.split == "dev" and "task_example" in case.tags
    } == set(TASK_EXAMPLES)
    assert {dept for case in cases for dept in case.expected} == set(Department)
    assert sum("injection" in case.tags for case in cases) >= 2
    assert sum("long" in case.tags for case in cases) >= 1
    assert all(len(case.message) <= MAX_MESSAGE_LENGTH for case in cases)


def test_valid_file_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, *_task_examples(), _case())

    cases = load_cases(path)

    assert [case.id for case in cases] == [
        "task-example-0",
        "task-example-1",
        "printer",
    ]
    assert cases[2].expected == (Department.HELP_DESK,)
    assert cases[2].tags == frozenset()


@pytest.mark.parametrize(
    ("bad_line", "match"),
    [
        (_case(id="task-example-0"), "duplicate id 'task-example-0'"),
        (_case(expected=["hr@example.com"]), "line 3"),
        (
            _case(expected=["help-desk@example.com", "it@example.com"]),
            "iff tagged 'ambiguous'",
        ),
        (_case(tags=["injection"]), "iff tagged 'injection'"),
        (
            _case(tags=["injection"], injected="help-desk@example.com"),
            "must not be one of the expected",
        ),
        (_case(message="   "), "must not be blank"),
        (_case(message="x" * (MAX_MESSAGE_LENGTH + 1)), "line 3"),
        (_case(tags=["long"]), "'long' case needs"),
        (_case(unknown_field=1), "line 3"),
        ('{"id": "broken", ', "line 3"),
    ],
    ids=[
        "duplicate_id",
        "unknown_department",
        "two_expected_without_ambiguous",
        "injection_without_injected",
        "injected_in_expected",
        "blank_message",
        "message_too_long",
        "long_tag_on_short_message",
        "extra_field",
        "malformed_json",
    ],
)
def test_invalid_file_is_rejected(
    bad_line: dict[str, Any] | str, match: str, tmp_path: Path
) -> None:
    path = _write(tmp_path, *_task_examples(), bad_line)

    with pytest.raises(DatasetError, match=match):
        load_cases(path)


@pytest.mark.parametrize(
    "lines",
    [
        [_task_examples()[0], _case()],
        [
            _task_examples()[0],
            _task_examples()[1] | {"split": "holdout"},
        ],
        [
            _task_examples()[0],
            _task_examples()[1] | {"message": TASK_EXAMPLES[1] + "!"},
        ],
    ],
    ids=["missing", "in_holdout", "not_verbatim"],
)
def test_task_examples_are_required_in_dev(
    lines: list[dict[str, Any]], tmp_path: Path
) -> None:
    path = _write(tmp_path, *lines)

    with pytest.raises(DatasetError, match="task example"):
        load_cases(path)
