"""CAD-Coder client (ADR-0112): a LOCAL, raw-CadQuery generator on the 4090.

Distinct from ``OllamaClient`` on purpose. ``OllamaClient`` constrains the model to a pydantic
schema via Ollama's ``format`` param -- right for a general instruct model asked for structured
DSL. CAD-Coder (``gudo7208/CAD-Coder``, Qwen2.5-7B, Apache-2.0) is a CODE model fine-tuned to emit
a CadQuery script from a part description; forcing a JSON schema on it fights its training. So this
client asks for raw text and returns the CadQuery source (fences stripped) for the exec bridge.

Per-part only, by design: CAD-Coder was trained on single sketch-extrude parts, never on repair or
assembly. It is a generator backend for ONE part; the decomposer feeds it parts and the assembler /
frontier-repair live elsewhere. ``cost_usd`` is always 0.0 (local inference -- the subscription-only
billing invariant holds).
"""

from __future__ import annotations

import os
import re

import httpx

from forge_mind.llm.client import LLMError
from forge_mind.llm.context_guard import check_not_truncated, plan_window
from forge_mind.llm.cost import LLMUsage

_DEFAULT_MODEL = "cad-coder"  # the local Ollama model imported from gudo7208/CAD-Coder (q8_0)
_DEFAULT_HOST = "http://localhost:11434"

# CAD-Coder emits a CadQuery script; the bridge expects the finished object bound to ``result``.
# CAD-Coder (Qwen2.5-7B-Instruct base, GRPO/CoT) is called EXACTLY as its model card prescribes:
# a bare user message wrapped in ChatML, NO system prompt (a system prompt derails the fine-tune --
# verified live: the same prompt with a system prompt stops after ~8 tokens; without one it emits a
# complete CadQuery script). We send raw ChatML via /api/generate (raw=true) so Ollama's auto-imported
# template / default "You are Qwen" system can't interfere.
_CHATML = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

_FENCE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# GREEDY decoding (temperature 0) degenerates on some prompts -- the model emits a bare "```" and stops
# (verified live). A modest sampling temperature escapes that path and yields a full script; the output
# is validated downstream, so nondeterminism is harmless. Env-overridable.
_DEFAULT_TEMP = 0.7

# Cap context so the KV cache stays small enough for VRAM co-residence with the other roster models.
_NUM_CTX = int(os.environ.get("FORGE_OLLAMA_NUM_CTX", "8192"))

# CAD-Coder's training prompts are imperative ("Create a ..."); a noun-phrase prompt makes degeneration
# more likely (verified: "A rectangular block ..." degenerated at temp 0, "Create a rectangular block
# ..." did not). Nudge non-imperative prompts toward the training distribution.
_IMPERATIVE_STARTS = ("create", "design", "make", "generate", "build", "model", "draw")


def _looks_like_cadquery(code: str) -> bool:
    """A real script imports/uses CadQuery; a degenerate one is empty or a bare fence marker."""
    return len(code) >= 20 and ("cadquery" in code or "Workplane" in code or "cq." in code)


def _normalize_prompt(prompt: str) -> str:
    p = prompt.strip()
    first = p.split(None, 1)[0].lower().rstrip(",.:") if p else ""
    if first in _IMPERATIVE_STARTS:
        return p
    return f"Create {p[0].lower()}{p[1:]}" if p else p


def _strip_fences(text: str) -> str:
    """Extract the CadQuery from the model's output. Prefers the contents of a fenced ```python block
    (CAD-Coder's CoT can precede the code); otherwise drops any stray fence lines (it often emits a
    trailing ``` with no opening fence)."""
    t = text.strip()
    m = _FENCE_BLOCK.search(t)
    if m:
        return m.group(1).strip()
    return "\n".join(ln for ln in t.splitlines() if not ln.strip().startswith("```")).strip()


class CadCoderClient:
    """Raw-CadQuery generation against a local Ollama server hosting CAD-Coder."""

    def __init__(
        self, model: str | None = None, host: str | None = None, timeout_s: float = 600.0
    ) -> None:
        self._model = model or os.environ.get("FORGE_CADCODER_MODEL", _DEFAULT_MODEL)
        self._host = (host or os.environ.get("FORGE_CADCODER_HOST", _DEFAULT_HOST)).rstrip("/")
        self._temp = float(os.environ.get("FORGE_CADCODER_TEMP", _DEFAULT_TEMP))
        self._http = httpx.Client(timeout=timeout_s)

    @property
    def model(self) -> str:
        return self._model

    def generate_cadquery(
        self, prompt: str, *, max_tokens: int = 2048, attempts: int = 3
    ) -> tuple[str, LLMUsage]:
        """Return ``(cadquery_source, usage)`` for a part description.

        The model occasionally DEGENERATES (emits a bare ``` and stops) on a given sample. Because we
        sample at temperature > 0, a retry draws a different completion, so we re-roll up to
        ``attempts`` times until the output actually looks like CadQuery. Raises ``LLMError`` on a
        transport failure or if every attempt degenerates (fail loud -- the caller records a failure).
        """
        user = _normalize_prompt(prompt)
        raw_prompt = _CHATML.format(prompt=user)
        num_ctx, num_predict = plan_window(
            prompt_chars=len(raw_prompt),
            requested_predict=max_tokens,
            floor=_NUM_CTX,
            label=f"CAD-Coder ({self._model})",
        )
        last = ""
        for _ in range(max(1, attempts)):
            try:
                resp = self._http.post(
                    f"{self._host}/api/generate",
                    json={
                        "model": self._model,
                        "raw": True,  # we supply the ChatML; no template / system injection
                        "prompt": raw_prompt,
                        "stream": False,
                        "options": {
                            "temperature": self._temp,
                            "num_predict": num_predict,
                            "num_ctx": num_ctx,
                        },
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise LLMError(
                    f"CAD-Coder request failed (host {self._host}, model {self._model}); is "
                    f"`ollama serve` running and the model pulled? {exc}"
                ) from exc
            data = resp.json()
            check_not_truncated(
                prompt_chars=len(raw_prompt),
                prompt_eval_count=int(data.get("prompt_eval_count") or 0),
                num_ctx=num_ctx,
                num_predict=num_predict,
                label=f"CAD-Coder ({self._model})",
            )
            code = _strip_fences(data.get("response", ""))
            last = code
            if _looks_like_cadquery(code):
                return code, LLMUsage(
                    model=self._model,
                    input_tokens=int(data.get("prompt_eval_count") or 0),
                    output_tokens=int(data.get("eval_count") or 0),
                    cost_usd=0.0,
                )
        raise LLMError(
            f"CAD-Coder degenerated on all {attempts} attempts (model {self._model}); "
            f"last output: {last[:80]!r}"
        )


# The grounding block exists because of a measured autopsy (ADR-0114): repairs failed by cutting
# where the solid ISN'T (a hole extruded downward under a part that spans z=0..h), by leaving cuts
# blind, and by inventing APIs. Each rule below kills one observed failure -- keep them evidence-bound.
_REPAIR_SYSTEM = """You are an expert CadQuery (Python) engineer fixing a part. You are given a \
CadQuery script and the validator's complaints about the part it produced. Return ONLY a corrected, \
complete CadQuery script -- keep what works, change the minimum needed to address the complaints. \
Import cadquery as cq and bind the final solid to a top-level variable named `result`. Millimeters.

COORDINATE FRAME (get this right first):
- `cq.Workplane("XY").rect(L, W).extrude(H)` spans x in [-L/2, L/2], y in [-W/2, W/2], z in [0, H].
- A cutting solid removes material ONLY where it overlaps the part. A cylinder extruded downward
  from z=0 sits BELOW a part that spans z in [0, H] and cuts nothing.

VERIFIED through-hole idiom (use this; do not invent methods):
    hole = (cq.Workplane("XY")
            .transformed(offset=(x, y, -1))     # start 1mm BELOW the bottom face
            .circle(d / 2)
            .extrude(H + 2))                    # end 1mm ABOVE the top face
    result = part.cut(hole)
The cutter must span PAST both faces so the hole exits both sides. For a horizontal hole use the
same idea on the "XZ" or "YZ" plane. Only call methods you are certain exist in CadQuery
(rect, circle, extrude, cut, union, transformed, translate, faces, workplane, hole, polygon).

Code only -- no explanation, no markdown fences."""


_CODER_SYSTEM = """You are an expert CadQuery (Python) engineer. Given a part description, output ONLY \
a complete CadQuery script that builds it and binds the final solid to a top-level variable named \
`result`. Import cadquery as cq. Millimeters. Build the WHOLE part in one script.

COORDINATE FRAME:
- cq.Workplane("XY").rect(L, W).extrude(H) spans x[-L/2, L/2], y[-W/2, W/2], z[0, H].
- cq.Workplane("XY").circle(r).extrude(H) is a cylinder on the z-axis spanning z[0, H].

Through-hole idiom (the cutter must pass BOTH faces so the hole exits both sides):
    hole = cq.Workplane("XY").transformed(offset=(x, y, -1)).circle(d/2).extrude(H+2)
    result = part.cut(hole)

Only call CadQuery methods you are certain exist (rect, circle, polygon, extrude, cut, union, translate,
transformed, faces, workplane). Code only -- no prose, no markdown fences."""


def generate_cadquery_coder(
    prompt: str,
    *,
    model: str | None = None,
    host: str | None = None,
    max_tokens: int = 2048,
    attempts: int = 2,
    temperature: float = 0.4,
    timeout_s: float = 600.0,
) -> tuple[str, LLMUsage]:
    """Generate a fresh CadQuery part with a strong GENERAL coder (default qwen3-coder), grounded by
    the coordinate-frame + idiom block (ADR-0114). Measured 10/10 on a block+hole where the CAD-Coder
    specialist scored 0/10 -- a strong instruction-follower beats a narrow one-shot fine-tune here.
    Retries past a degenerate draft; raises ``LLMError`` if every attempt is unusable."""
    resolved_model = (
        model
        or os.environ.get("FORGE_CODER_MODEL", "")
        or os.environ.get("FORGE_OLLAMA_MODEL", "qwen3-coder:30b")
    )
    resolved_host = (host or os.environ.get("FORGE_OLLAMA_HOST", _DEFAULT_HOST)).rstrip("/")
    prompt_chars = len(_CODER_SYSTEM) + len(prompt)
    num_ctx, num_predict = plan_window(
        prompt_chars=prompt_chars,
        requested_predict=max_tokens,
        floor=_NUM_CTX,
        label=f"coder ({resolved_model})",
    )
    http = httpx.Client(timeout=timeout_s)
    last = ""
    for _ in range(max(1, attempts)):
        try:
            resp = http.post(
                f"{resolved_host}/api/chat",
                json={
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": _CODER_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict,
                        "num_ctx": num_ctx,
                    },
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"coder request failed (host {resolved_host}, model {resolved_model}): {exc}"
            ) from exc
        data = resp.json()
        check_not_truncated(
            prompt_chars=prompt_chars,
            prompt_eval_count=int(data.get("prompt_eval_count") or 0),
            num_ctx=num_ctx,
            num_predict=num_predict,
            label=f"coder ({resolved_model})",
        )
        code = _strip_fences((data.get("message") or {}).get("content", ""))
        last = code
        if _looks_like_cadquery(code):
            return code, LLMUsage(
                model=resolved_model,
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
                cost_usd=0.0,
            )
    raise LLMError(
        f"coder produced no usable script in {attempts} attempts "
        f"(model {resolved_model}); last: {last[:80]!r}"
    )


def repair_cadquery(
    previous_code: str,
    feedback: str,
    object_name: str,
    *,
    model: str | None = None,
    host: str | None = None,
    max_tokens: int = 2048,
    attempts: int = 2,
    timeout_s: float = 600.0,
) -> tuple[str, LLMUsage]:
    """Repair a CadQuery script in CadQuery (ADR-0114). The coder writes CadQuery, so the REPAIR must
    speak CadQuery too -- handing CadQuery to a build123d prompt is the language mismatch that made
    every revision fail to build. Uses the local repair model (qwen2.5-coder, an instruct model, so a
    normal system+user chat -- unlike CAD-Coder which needs raw ChatML). $0. Retries past a degenerate
    draft; raises ``LLMError`` if every attempt is unusable."""
    resolved_model = model or os.environ.get("FORGE_OLLAMA_MODEL", "qwen3-coder:30b")
    resolved_host = (host or os.environ.get("FORGE_OLLAMA_HOST", _DEFAULT_HOST)).rstrip("/")
    user = (
        f"The part should be: {object_name}.\n\n"
        f"CURRENT CadQuery script:\n```python\n{previous_code}\n```\n\n"
        f"The validator reported these problems:\n{feedback}\n\n"
        "Return the corrected CadQuery script."
    )
    prompt_chars = len(_REPAIR_SYSTEM) + len(user)
    num_ctx, num_predict = plan_window(
        prompt_chars=prompt_chars,
        requested_predict=max_tokens,
        floor=_NUM_CTX,
        label=f"CadQuery repair ({resolved_model})",
    )
    http = httpx.Client(timeout=timeout_s)
    last = ""
    for _ in range(max(1, attempts)):
        try:
            resp = http.post(
                f"{resolved_host}/api/chat",
                json={
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": _REPAIR_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": num_predict, "num_ctx": num_ctx},
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"CadQuery repair request failed (host {resolved_host}, model {resolved_model}): {exc}"
            ) from exc
        data = resp.json()
        check_not_truncated(
            prompt_chars=prompt_chars,
            prompt_eval_count=int(data.get("prompt_eval_count") or 0),
            num_ctx=num_ctx,
            num_predict=num_predict,
            label=f"CadQuery repair ({resolved_model})",
        )
        code = _strip_fences((data.get("message") or {}).get("content", ""))
        last = code
        if _looks_like_cadquery(code):
            return code, LLMUsage(
                model=resolved_model,
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
                cost_usd=0.0,
            )
    raise LLMError(
        f"CadQuery repair produced no usable script in {attempts} attempts "
        f"(model {resolved_model}); last: {last[:80]!r}"
    )


__all__ = ["CadCoderClient", "generate_cadquery_coder", "repair_cadquery"]
