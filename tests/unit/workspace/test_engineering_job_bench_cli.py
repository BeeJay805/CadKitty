"""The benchmark must expose the same optional budgets as workspace execution."""

import pytest
from scripts.workspace_engineering_job_bench import parse_args

REQUIRED = ["--brief", "brief.txt", "--output", "run"]


def test_omitted_budgets_do_not_create_a_cutoff() -> None:
    args = parse_args(REQUIRED)
    assert args.max_actions is None
    assert args.max_seconds is None


def test_explicit_budgets_above_legacy_caps_are_preserved() -> None:
    args = parse_args([*REQUIRED, "--max-actions", "128", "--max-seconds", "14400"])
    assert args.max_actions == 128
    assert args.max_seconds == 14400


@pytest.mark.parametrize("option", ["--max-actions", "--max-seconds"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_explicit_budget_is_rejected(option: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args([*REQUIRED, option, value])
    assert exc.value.code == 2
