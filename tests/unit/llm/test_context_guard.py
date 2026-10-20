"""ADR-0115 context-window guard: the 2026-07-15 RCA failure (a 28-29K-char author prompt silently
clipped to 4098 evaluated tokens by Ollama) must be LOUD -- refused pre-flight when overflow is
certain, and raised post-hoc when the server's own prompt_eval_count carries the truncation
signature. A prefix-cached repeat (tiny prompt_eval_count, far below the window cap) must NOT fire."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from forge_mind.llm.client import LLMError
from forge_mind.llm.context_guard import (
    check_fits,
    check_not_truncated,
    fit_num_ctx,
    guard_enabled,
    min_prompt_tokens,
    plan_window,
)
from forge_mind.llm.ollama_client import OllamaClient


class _Brief(BaseModel):
    object_name: str


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_min_prompt_tokens_is_a_lower_bound() -> None:
    # 28K chars (the RCA prompt) lower-bounds to ~4667 tokens -- above the 4098 the server reported.
    assert min_prompt_tokens(28_000) == 4666
    assert min_prompt_tokens(0) == 0


def test_check_fits_raises_on_certain_overflow() -> None:
    # RCA shape: ~28K chars + a 4096 generation budget cannot fit an 8192 window.
    with pytest.raises(LLMError, match="cannot fit"):
        check_fits(prompt_chars=28_000, num_ctx=8192, num_predict=4096, label="t")


def test_check_fits_passes_a_normal_prompt() -> None:
    check_fits(prompt_chars=8_000, num_ctx=8192, num_predict=2048, label="t")


def test_check_not_truncated_fires_on_the_rca_signature() -> None:
    # Server evaluated 4098 tokens of a 28K-char prompt with the count pinned at the window cap.
    with pytest.raises(LLMError, match="TRUNCATED"):
        check_not_truncated(
            prompt_chars=28_000,
            prompt_eval_count=4098,
            num_ctx=8192,
            num_predict=4096,
            label="t",
        )


def test_check_not_truncated_fires_on_the_live_probe_signature() -> None:
    # MEASURED live (2026-07-18): 69,599 chars at num_ctx=8192 / num_predict=64 -> the server
    # evaluated 4098 tokens (Ollama clips to ~num_ctx/2, NOT to num_ctx - num_predict). The guard
    # must catch truncation even when the generation budget is small.
    with pytest.raises(LLMError, match="TRUNCATED"):
        check_not_truncated(
            prompt_chars=69_599,
            prompt_eval_count=4098,
            num_ctx=8192,
            num_predict=64,
            label="t",
        )


def test_check_not_truncated_ignores_a_prefix_cached_repeat() -> None:
    # A cached re-roll legitimately reports a tiny evaluated count FAR below the cap -- no fire.
    check_not_truncated(
        prompt_chars=28_000, prompt_eval_count=12, num_ctx=8192, num_predict=4096, label="t"
    )


def test_check_not_truncated_ignores_missing_server_count() -> None:
    check_not_truncated(
        prompt_chars=28_000, prompt_eval_count=0, num_ctx=8192, num_predict=4096, label="t"
    )


def test_check_not_truncated_passes_an_honest_short_prompt() -> None:
    check_not_truncated(
        prompt_chars=2_000, prompt_eval_count=520, num_ctx=8192, num_predict=2048, label="t"
    )


def test_kill_switch_disables_both_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_OLLAMA_CTX_GUARD", "0")
    assert not guard_enabled()
    check_fits(prompt_chars=1_000_000, num_ctx=1024, num_predict=1024, label="t")
    check_not_truncated(
        prompt_chars=1_000_000, prompt_eval_count=900, num_ctx=1024, num_predict=64, label="t"
    )


def test_fit_num_ctx_small_prompt_stays_at_the_floor() -> None:
    assert fit_num_ctx(prompt_chars=2_000, num_predict=2048, floor=8192) == 8192


def test_fit_num_ctx_grows_for_the_measured_author_prompt() -> None:
    # The author system prompt measured 8,004 true tokens (28,192 chars) -- with a brief and a
    # 4096 generation budget the window must grow past the old fixed 8192 that clipped it.
    assert fit_num_ctx(prompt_chars=30_000, num_predict=4096, floor=8192) == 16384


def test_fit_num_ctx_rounds_up_in_steps() -> None:
    # needed = 24000/3 + 2048 + 64 = 10112 -> next 4096 step = 12288 (stable across similar calls).
    assert fit_num_ctx(prompt_chars=24_000, num_predict=2048, floor=8192) == 12288


def test_fit_num_ctx_respects_the_env_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_OLLAMA_MAX_CTX", "8192")
    assert fit_num_ctx(prompt_chars=30_000, num_predict=4096, floor=8192) == 8192


def test_plan_window_clamps_the_cloud_sized_author_budget() -> None:
    # The production author default is max_tokens=16000 (cloud-sized). Locally that budget must be
    # CLAMPED, not refused: the 29K-char author prompt + a 4096 cap fit a 16384 window fine.
    # (This exact shape hard-blocked arm B of the 2026-07-18 A/B before the clamp.)
    num_ctx, num_predict = plan_window(
        prompt_chars=29_234, requested_predict=16_000, floor=8192, label="t"
    )
    assert num_ctx == 16384
    assert num_predict == 6144


def test_plan_window_raises_when_even_the_minimum_budget_cannot_fit() -> None:
    with pytest.raises(LLMError, match="cannot fit"):
        plan_window(prompt_chars=120_000, requested_predict=4096, floor=8192, label="t")


def test_plan_window_kill_switch_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_OLLAMA_CTX_GUARD", "0")
    num_ctx, num_predict = plan_window(
        prompt_chars=1_000_000, requested_predict=16_000, floor=8192, label="t"
    )
    assert num_ctx == 16384  # still capped
    assert num_predict == 6144  # still budget-capped -- only the raise is disabled


def test_ollama_client_preflights_a_prompt_beyond_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even the grown-to-cap window (16384) cannot fit ~120K chars + 4096 predict; the client must
    # refuse BEFORE the HTTP call -- a post here would prove the guard didn't run.
    c = OllamaClient()

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("HTTP call should not happen on certain overflow")

    monkeypatch.setattr(c._http, "post", _boom)
    with pytest.raises(LLMError, match="cannot fit"):
        c.generate_structured(
            system="s" * 120_000, user="u", output_model=_Brief, model="m", max_tokens=4096
        )


def test_ollama_client_grows_num_ctx_and_passes_a_fitting_big_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 30K chars used to be silently clipped at the fixed 8192; now the request must carry a grown
    # window and an honest (un-clipped) server count passes.
    posted: dict[str, Any] = {}
    resp = _FakeResp(
        {
            "message": {"content": '{"object_name": "ok"}'},
            "prompt_eval_count": 8600,  # plausible true count for ~30K chars -- no truncation
            "eval_count": 10,
        }
    )

    def _post(url: str, json: dict[str, Any]) -> _FakeResp:
        posted.update(json)
        return resp

    c = OllamaClient()
    monkeypatch.setattr(c._http, "post", _post)
    out, _ = c.generate_structured(
        system="s" * 30_000, user="u", output_model=_Brief, model="m", max_tokens=4096
    )
    assert out.object_name == "ok"
    assert posted["options"]["num_ctx"] == 16384


def test_ollama_client_raises_on_truncated_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ~90K-char prompt passes pre-flight against the 16384 cap (lower bound ~15K + 512), but its
    # true count (~25K tokens) cannot fit -- the server clips to ~num_ctx/2 = 8192 and the client
    # must raise instead of returning output built from a clipped prompt.
    resp = _FakeResp(
        {
            "message": {"content": '{"object_name": "clipped"}'},
            "prompt_eval_count": 8192,
            "eval_count": 10,
        }
    )
    c = OllamaClient()
    monkeypatch.setattr(c._http, "post", lambda *a, **k: resp)
    with pytest.raises(LLMError, match="TRUNCATED"):
        c.generate_structured(
            system="s" * 90_000, user="u", output_model=_Brief, model="m", max_tokens=512
        )
