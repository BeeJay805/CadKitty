"""Subscription-billed LLM provider: headless Claude Code as a drop-in client.

``ClaudeCodeClient`` duck-types the two ``LLMClient`` methods the engine uses
(``generate_structured`` / ``generate_structured_with_image``) but shells out to
``claude -p`` (headless Claude Code). Calls bill the user's Claude SUBSCRIPTION
(plan rate limits), not the metered ANTHROPIC_API_KEY -- the point is cheap
iteration: bench reruns, smoke loops, breadth scoring, without per-token spend.

HONESTY BOUNDARIES (read before trusting numbers from this provider):
  * Results are labeled ``provider=claude_code`` and are NOT directly comparable
    to API-path runs for production swap decisions: the headless wrapper adds its
    own system context, structured output is schema-in-prompt + parse (not the
    API's forced tool-use parse), and per-call ``max_tokens`` is NOT enforced --
    truncation semantics differ from the production engine.
  * ``LLMUsage.cost_usd`` is recorded as 0.0 (subscription): the loop's
    dollar-budget stop never fires, so runs are bounded ONLY by iteration caps
    and the bench's trial caps. Token counts are still recorded from the CLI's
    usage envelope when present.
  * Heavy benches consume the subscription's rate-limit windows; expect
    throttling on long runs, which appears as LLMError after retries.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage

T = TypeVar("T", bound=BaseModel)

# 600 starved the chief's REVISION call (row 56: 12 build cards + rules + the assembled
# render on the brain model exceeds 10 min on the subscription path -> TimeoutExpired ->
# the repair silently no-fired). 1200 bounds the slowest legitimate call, not the typical.
_CALL_TIMEOUT_S = 1200
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _claude_exe() -> str:
    for name in ("claude", "claude.cmd", "claude.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise LLMError("claude CLI not found on PATH (needed for the subscription provider)")


def _json_payload(text: str) -> str:
    """The model's JSON, tolerant of a stray code fence or prose preamble."""
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _neutral_cwd() -> Path:
    """An EMPTY scratch directory the ``claude -p`` child runs in.

    Without this, the child inherits the ENGINE's cwd -- a Forge-Mind checkout -- and
    every generation call silently loads that repo's CLAUDE.md, hooks, and MCP config
    as session context. Measured consequences: structured-output calls sometimes answer
    as a repo assistant instead of emitting pure JSON (the v2b/v2c decompose deaths --
    NONDETERMINISTIC, because the repo state the context reflects changes between runs),
    plus a large per-call latency/token tax. A neutral cwd gives every engine call a
    clean, repo-free session; user-level config still applies (provider-paired either way).
    """
    d = Path(tempfile.gettempdir()) / "forge_mind_claude_code_neutral"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ClaudeCodeClient:
    """LLMClient-compatible provider that runs prompts through ``claude -p``."""

    def __init__(self, *, timeout_s: int = _CALL_TIMEOUT_S) -> None:
        self._exe = _claude_exe()
        self._timeout_s = timeout_s
        self._cwd = _neutral_cwd()

    # ---- core subprocess call ------------------------------------------------------

    def _run(self, prompt: str, model: str, allowed_tools: str | None) -> tuple[str, LLMUsage]:
        cmd = [self._exe, "-p", "--output-format", "json", "--model", model]
        # Latency lever: Opus 4.8 runs adaptive thinking at effort=high by default, and on the
        # subscription that thinking IS the output-token/latency bottleneck. FORGE_CLAUDE_EFFORT
        # (e.g. "low") trims it; unset -> no flag -> the CLI default (byte-identical).
        effort = os.environ.get("FORGE_CLAUDE_EFFORT", "").strip()
        if effort:
            cmd += ["--effort", effort]
        if allowed_tools:
            cmd += ["--allowedTools", allowed_tools]
        env = dict(os.environ)
        # Never let the subprocess fall back to metered API billing: the whole point
        # of this provider is the subscription path. (The scrubbed env was previously
        # built but NOT passed to subprocess.run -- the inherited key made the CLI 401
        # and could have billed the metered key; the i4 probe's failure field caught it.)
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        try:
            proc = subprocess.run(
                cmd,
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=self._timeout_s,
                cwd=str(self._cwd),  # NEUTRAL dir: no repo CLAUDE.md/hooks/MCP in the session
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude -p timed out after {self._timeout_s}s") from exc
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise LLMError(f"claude -p failed (rc={proc.returncode}): {err[:400] or out[:400]}")
        try:
            envelope = json.loads(out)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude -p emitted a non-JSON envelope: {out[:300]}") from exc
        if envelope.get("is_error"):
            raise LLMError(f"claude -p reported an error: {str(envelope)[:400]}")
        result = str(envelope.get("result", ""))
        u = envelope.get("usage") or {}
        usage = LLMUsage(
            model=model,
            input_tokens=int(u.get("input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_read_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
            cost_usd=0.0,  # subscription-billed: see module docstring
        )
        if os.environ.get("FORGE_DEBUG_STRUCTURED") == "1":
            details = u.get("output_tokens_details") or {}
            print(
                f"[run] model={model} effort={effort or 'default'} "
                f"out_tokens={usage.output_tokens} thinking={details.get('thinking_tokens')}",
                file=sys.stderr,
                flush=True,
            )
        return result, usage

    def _structured(
        self,
        *,
        prompt_head: str,
        output_model: type[T],
        model: str,
        allowed_tools: str | None,
    ) -> tuple[T, LLMUsage]:
        schema = json.dumps(output_model.model_json_schema(), indent=None)
        ask = (
            f"{prompt_head}\n\n"
            "Respond with ONLY one JSON object (no prose, no markdown fence) that "
            f"validates against this JSON schema:\n{schema}"
        )
        total = LLMUsage(model=model, input_tokens=0, output_tokens=0, cost_usd=0.0)
        last: Exception | None = None
        prompt = ask
        for _attempt in range(2):
            text, usage = self._run(prompt, model, allowed_tools)
            total = LLMUsage(
                model=model,
                input_tokens=total.input_tokens + usage.input_tokens,
                output_tokens=total.output_tokens + usage.output_tokens,
                cache_read_tokens=total.cache_read_tokens + usage.cache_read_tokens,
                cache_write_tokens=total.cache_write_tokens + usage.cache_write_tokens,
                cost_usd=0.0,
            )
            if os.environ.get("FORGE_DEBUG_STRUCTURED") == "1":
                print(
                    f"[structured] {output_model.__name__} attempt {_attempt + 1}: "
                    f"out_tokens={usage.output_tokens} resp_chars={len(text)}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                parsed = output_model.model_validate_json(_json_payload(text))
                return parsed, total
            except (ValidationError, ValueError) as exc:
                last = exc
                prompt = (
                    f"{ask}\n\nYour previous answer did not validate "
                    f"({str(exc)[:300]}). Output ONLY the corrected JSON object."
                )
        raise LLMError(
            f"structured output failed to parse twice (claude_code provider, model {model}): {last}"
        ) from last

    # ---- LLMClient-compatible surface ---------------------------------------------

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[T, LLMUsage]:
        del max_tokens, cache_system  # not enforceable through claude -p (see docstring)
        head = f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER REQUEST:\n{user}"
        return self._structured(
            prompt_head=head, output_model=output_model, model=model, allowed_tools=None
        )

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: type[T],
        model: str,
        max_tokens: int,
    ) -> tuple[T, LLMUsage]:
        del max_tokens
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(image_png)
            png_path = Path(fh.name)
        try:
            head = (
                f"SYSTEM INSTRUCTIONS:\n{system}\n\n"
                f"First use the Read tool to view the image at: {png_path}\n\n"
                f"USER REQUEST:\n{user}"
            )
            return self._structured(
                prompt_head=head,
                output_model=output_model,
                model=model,
                allowed_tools="Read",
            )
        finally:
            png_path.unlink(missing_ok=True)


def subscription_client_if_selected() -> ClaudeCodeClient | None:
    """The bench scripts' provider switch: a ClaudeCodeClient when
    ``FORGE_LLM_PROVIDER=claude_code``, else None (the engine's default API client)."""
    if os.environ.get("FORGE_LLM_PROVIDER") == "claude_code":
        return ClaudeCodeClient()
    return None


__all__ = ["ClaudeCodeClient", "subscription_client_if_selected"]
