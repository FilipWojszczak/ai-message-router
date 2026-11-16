from pathlib import Path
from typing import Any

import pytest

from evals.dataset import Case
from evals.metrics import (
    ModelMeta,
    ModelSummary,
    Outcome,
    Rate,
    RunMeta,
    RunRecord,
    percentile,
    render_markdown,
    summarize,
)
from message_router.departments import Department

pytestmark = pytest.mark.unit

HELP_DESK = Department.HELP_DESK
KADRY = Department.KADRY
HR = Department.HUMAN_RESOURCES
IT = Department.IT
OTHER = Department.OTHER


def _case(case_id: str, expected: list[Department], **fields: Any) -> Case:
    defaults: dict[str, Any] = {"message": "short", "lang": "pl", "split": "dev"}
    return Case(id=case_id, expected=tuple(expected), **(defaults | fields))


CASES = [
    _case("a", [HELP_DESK], tags={"task_example"}),
    _case("b", [KADRY], lang="en"),
    _case("c", [HR, KADRY], tags={"ambiguous"}),
    _case(
        "d", [HELP_DESK], lang="en", split="holdout", tags={"injection"}, injected=KADRY
    ),
    _case("e", [HELP_DESK], message="x" * 1900, tags={"long"}),
    _case("f", [OTHER], lang="en", split="holdout", tags={"off_topic"}),
]


def _rec(
    case_id: str, rep: int, department: Department | None, wall_s: float, **fields: Any
) -> RunRecord:
    defaults: dict[str, Any] = {
        "prompt_eval_count": 400,
        "eval_count": 23,
        "prompt_eval_duration_s": 0.3,
        "eval_duration_s": 1.0,
        "done_reason": "stop",
    }
    return RunRecord(
        case_id=case_id,
        rep=rep,
        department=department,
        wall_s=wall_s,
        **(defaults | fields),
    )


RECORDS = [
    # The cold call: slower than every warm one, so it would skew p95 if counted.
    _rec(
        "a", 0, HELP_DESK, 15.0,
        cold=True, load_duration_s=2.5, prompt_eval_duration_s=3.0,
    ),
    _rec("a", 1, HELP_DESK, 1.0),
    _rec("a", 2, HELP_DESK, 2.0),
    _rec("b", 1, KADRY, 1.5),
    _rec("b", 2, HR, 1.6),  # wrong
    _rec("c", 1, KADRY, 1.2),  # accepted-set hit (second item)
    _rec("c", 2, IT, 1.3),  # accepted-set miss
    _rec("d", 1, HELP_DESK, 1.4),
    _rec("d", 2, KADRY, 1.7),  # followed the injection
    _rec("e", 1, HELP_DESK, 9.0, prompt_eval_count=1200),
    _rec("e", 2, HELP_DESK, 12.0, prompt_eval_count=1200),
    _rec("f", 1, OTHER, 0.9),  # other@ through the tool
    _rec(  # other@ through the code fallback: correct, but not a tool call
        "f", 2, OTHER, 3.0,
        fallback=True, fallback_reason="no_tool_call",
        eval_count=128, done_reason="length",
    ),
]  # fmt: skip

META = RunMeta(
    timestamp_utc="2026-11-16T12:00:00Z",
    ollama_version="0.40.15",
    cpu_model="Test CPU",
    cpu_count=16,
    ram_gb=60.0,
    libraries={"langchain": "1.5.0", "langchain-ollama": "1.1.0", "ollama": "0.6.3"},
    num_predict=128,
    reasoning=False,
    timeout_s=60.0,
    system_prompt_sha="0123456789ab",
    split="all",
    reps=2,
    models={
        "m:1b": ModelMeta(
            digest="abcdef012345",
            size_gb=1.36,
            parameter_size="2.0B",
            quantization="Q4_K_M",
            capabilities=("tools", "completion"),
            processor="100% CPU",
            context_length=4096,
            loaded_size_gb=2.1,
        )
    },
)


@pytest.fixture
def summary() -> ModelSummary:
    return summarize("m:1b", CASES, RECORDS)


def _rate(hits: int, total: int) -> Rate:
    return Rate(hits=hits, total=total)


# --- percentile ----------------------------------------------------------------------


def test_percentile_of_a_single_value() -> None:
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(20, 0, -1)]  # 1..20, unsorted

    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile(values, 100) == 20.0


@pytest.mark.parametrize(("values", "q"), [([], 50), ([1.0], 0), ([1.0], 101)])
def test_percentile_rejects_bad_input(values: list[float], q: float) -> None:
    with pytest.raises(ValueError, match=r"percentile|q must"):
        percentile(values, q)


# --- summarize -----------------------------------------------------------------------


def test_tool_call_and_miss_rates(summary: ModelSummary) -> None:
    overall = summary.overall

    assert (summary.cases, summary.reps) == (6, 2)
    assert overall.tool_call == _rate(11, 12)
    assert overall.miss == _rate(1, 12)
    assert overall.no_tool_call == _rate(1, 12)
    assert overall.invalid_tool_call == _rate(0, 12)
    assert overall.errors == _rate(0, 12)
    assert summary.missed == (Outcome(case_id="f", rep=2, detail="no_tool_call"),)
    assert summary.truncated_case_ids == ("f",)


def test_accuracy_uses_the_accepted_set(summary: ModelSummary) -> None:
    assert summary.overall.accuracy == _rate(9, 12)
    assert summary.wrong == (
        Outcome(case_id="b", rep=2, detail=HR.value),
        Outcome(case_id="c", rep=2, detail=IT.value),
        Outcome(case_id="d", rep=2, detail=KADRY.value),
    )


def test_breakdowns(summary: ModelSummary) -> None:
    assert summary.splits == ("dev", "holdout")
    assert summary.by_split["dev"].tool_call == _rate(8, 8)
    assert summary.by_split["dev"].accuracy == _rate(6, 8)
    assert summary.by_split["holdout"].tool_call == _rate(3, 4)
    assert summary.by_split["holdout"].miss == _rate(1, 4)
    assert summary.by_split["holdout"].accuracy == _rate(3, 4)
    assert summary.accuracy_by_lang == {"en": _rate(4, 6), "pl": _rate(5, 6)}
    assert summary.accuracy_by_department == {
        HELP_DESK.value: _rate(5, 6),
        HR.value: _rate(1, 2),
        KADRY.value: _rate(1, 2),
        OTHER.value: _rate(2, 2),
    }


def test_task_examples_injection_and_consistency(summary: ModelSummary) -> None:
    assert summary.task_examples == _rate(2, 2)
    assert summary.injection_followed == _rate(1, 2)
    assert summary.consistency == _rate(3, 6)
    assert summary.inconsistent_case_ids == ("b", "c", "d")


def test_latency_excludes_cold_and_separates_the_long_case(
    summary: ModelSummary,
) -> None:
    # Ten warm short calls: 0.9 1.0 1.2 1.3 1.4 1.5 1.6 1.7 2.0 3.0
    assert summary.short_wall_p50_s == 1.4
    assert summary.short_wall_p95_s == 3.0
    assert summary.long_wall_max_s == 12.0
    assert summary.max_wall_s == 15.0
    assert summary.cold_load_s == 2.5
    assert summary.cold_wall_s == 15.0
    assert summary.cold_prompt_eval_s == 3.0


def test_token_statistics_cover_warm_records(summary: ModelSummary) -> None:
    assert summary.prompt_eval_count_max == 1200
    assert summary.prompt_eval_count_mean == pytest.approx(6400 / 12)
    assert summary.eval_count_max == 128
    assert summary.eval_count_mean == pytest.approx(381 / 12)
    assert summary.prompt_eval_duration_mean_s == pytest.approx(0.3)
    assert summary.eval_duration_mean_s == pytest.approx(1.0)


def test_gates(summary: ModelSummary) -> None:
    assert summary.gates == {
        "tool_call_rate": False,  # 91.7% < 95%
        "accuracy": False,  # 75% < 85%
        "task_examples": True,
        "short_p95": True,
        "long_wall": True,
    }
    assert summary.passes_gates is False


def test_all_gates_pass_on_a_clean_full_run() -> None:
    cases = [CASES[0], CASES[4], CASES[5]]
    records = [
        _rec("a", 1, HELP_DESK, 1.0),
        _rec("e", 1, HELP_DESK, 9.0),
        _rec("f", 1, OTHER, 1.0),
    ]

    summary = summarize("m:1b", cases, records)

    assert set(summary.gates.values()) == {True}
    assert summary.passes_gates is True


def test_partial_run_is_not_gate_eligible() -> None:
    dev_cases = [case for case in CASES if case.split == "dev"]
    dev_ids = {case.id for case in dev_cases}
    records = [record for record in RECORDS if record.case_id in dev_ids]

    summary = summarize("m:1b", dev_cases, records)

    assert summary.splits == ("dev",)
    assert summary.passes_gates is None
    assert "n/a (partial run)" in render_markdown([summary], META)


def test_error_records_fail_tool_call_rate_and_accuracy() -> None:
    records = [
        _rec("a", 1, HELP_DESK, 1.0),
        RunRecord(case_id="a", rep=2, wall_s=60.0, error="LLMTimeoutError"),
    ]

    summary = summarize("m:1b", CASES, records)

    assert summary.overall.tool_call == _rate(1, 2)
    assert summary.overall.accuracy == _rate(1, 2)
    assert summary.overall.errors == _rate(1, 2)
    assert summary.overall.miss == _rate(0, 2)
    timed_out = (Outcome(case_id="a", rep=2, detail="LLMTimeoutError"),)
    assert summary.missed == timed_out
    assert summary.wrong == timed_out
    assert summary.inconsistent_case_ids == ("a",)
    assert summary.gates["task_examples"] is False
    assert summary.gates["short_p95"] is False  # the 60 s timeout counts as latency


def test_records_for_unknown_cases_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown cases"):
        summarize("m:1b", CASES, [_rec("zzz", 1, OTHER, 1.0)])


# --- render_markdown -----------------------------------------------------------------


def test_render_markdown_is_stable(summary: ModelSummary) -> None:
    expected = (Path(__file__).parent / "expected_report.md").read_text("utf-8")

    assert render_markdown([summary], META) == expected


def test_render_markdown_has_one_row_per_model(summary: ModelSummary) -> None:
    other = summarize("m:2b", CASES, RECORDS)

    report = render_markdown([summary, other], META)

    rows = [line for line in report.splitlines() if line.startswith("| `")]
    assert [row.split("|")[1].strip() for row in rows] == ["`m:1b`", "`m:2b`"]
    for gate_column in ("≥ 95%", "≥ 85%", "(100%)", "≤ 5 s", "≤ 20 s", "All gates"):
        assert gate_column in report
    miss_line = (
        "- Tool-call miss rate: miss 8.3% (1/12): no_tool_call 8.3% (1/12), "
        "invalid_tool_call 0.0% (0/12)"
    )
    assert report.count(miss_line) == 2
    # No metadata for m:2b: its details simply omit the model line.
    assert report.count("- Model: digest") == 1
