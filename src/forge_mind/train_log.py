"""Training-data tap (the 4090 flywheel's intake, founder-directed 2026-07-12).

Every part the engine authors is a candidate training pair for a local CAD-author
model: (prompt -> build123d code) labeled by the validator's verdict -- the reward
signal the field lacks, used as a teacher. This module appends one JSONL row per
authored part at pipeline end; a separate backfill script harvests historical runs
from their on-disk iteration artifacts into the same shape.

FAIL-OPEN and out of the hot path: logging can never break or slow a run in any way
that matters (one file append). FORGE_TRAIN_LOG=0 disables; FORGE_TRAIN_LOG_PATH
overrides the destination (default ~/.forge-train/parts.jsonl -- outside the repo,
so datasets never bloat git).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _enabled() -> bool:
    return os.environ.get("FORGE_TRAIN_LOG", "1").strip().lower() not in ("0", "false", "no")


def _path() -> Path:
    override = os.environ.get("FORGE_TRAIN_LOG_PATH", "").strip()
    return Path(override) if override else Path.home() / ".forge-train" / "parts.jsonl"


def _write_row(row: dict[str, Any]) -> bool:
    """Fail-open JSONL append (one row per authored/selected part). Never raises."""
    try:
        if not _enabled():
            return False
        out = _path()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def _coder_provider() -> str:
    """The coder role's provider, resolved the same way the factory does (role env, then
    the global switch, then the metered default)."""
    return (
        os.environ.get("FORGE_PROVIDER_CODER", "").strip()
        or os.environ.get("FORGE_LLM_PROVIDER", "").strip()
        or "anthropic"
    )


def _label_env() -> dict[str, Any]:
    """Validator and author identity attached to every newly written row."""
    from forge_mind.validation_provenance import validation_fingerprint

    provider = _coder_provider()
    return {
        "validation_fingerprint": validation_fingerprint(),
        "feature_presence": os.environ.get("FORGE_FEATURE_PRESENCE", "") == "1",
        "vision_model": os.environ.get("FORGE_VISION_MODEL", "qwen2.5vl:7b"),
        "feature_plan": os.environ.get("FORGE_FEATURE_PLAN", "") == "1",
        "strict_disposition": os.environ.get("FORGE_STRICT_DISPOSITION", "1") != "0",
        "coder_provider": provider,
        "coder_model": (
            os.environ.get("FORGE_OLLAMA_MODEL", "")
            if provider == "ollama"
            else os.environ.get("FORGE_AUTHOR_MODEL", "")
        ),
        "critic_refutation_override": os.environ.get("FORGE_CRITIC_REFUTATION_OVERRIDE", "") == "1",
    }


def log_part(
    lr: Any,
    material: Any = None,
    *,
    source: str = "live",
    provenance: str = "agentic",
    spec: dict[str, Any] | None = None,
    checks: dict[str, str] | None = None,
    claimable: bool | None = None,
) -> bool:
    """Append one training row for a finished part run. Returns True iff a row was
    written (False = disabled, no code to log, or any error -- never raises).

    The T6 flywheel labels are ADDITIVE (old readers unaffected -- new keys, same existing
    keys): ``provenance`` (agentic|analytic|topopt -- WHICH author produced the part, the
    method-SELECTION signal), ``spec`` (the RequirementSpec dump when a typed spec drove it),
    ``checks`` (the uniform gate's per-check verdicts), and ``claimable``. The agentic path
    carries code + provenance='agentic'; the physics authors carry no build123d code and log
    their selection row via ``log_selection`` instead."""
    code = getattr(lr, "final_code", None)
    if not code:
        return False  # author_failed / brief_failed runs carry no code to learn from
    validation = getattr(lr, "validation", None)
    brief = getattr(lr, "brief", None)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "provenance": provenance,
        "prompt": getattr(lr, "prompt", ""),
        "object_name": getattr(brief, "object_name", None),
        "material": getattr(material, "name", None) or getattr(brief, "material", None),
        "code": code,
        "code_sha": hashlib.sha256(code.encode()).hexdigest()[:16],
        # the labels a rejection-sampling / DPO pipeline filters on:
        "accepted": getattr(lr, "stopped_reason", "") == "accepted",
        "geometry_passed": bool(getattr(lr, "geometry_passed", False)),
        "hard_ok": bool(validation.hard_ok) if validation is not None else None,
        "critic_approved": bool(getattr(lr, "critic_approved", False)),
        "stopped_reason": getattr(lr, "stopped_reason", None),
        "iterations": len(getattr(lr, "iterations", []) or []),
        # T6 additive flywheel labels (None on the agentic path -- its spec/gate are implicit):
        "spec": spec,
        "checks": checks,
        "claimable": claimable,
        # Label provenance (ADR-0117 additive keys): WHICH validator/critic configuration
        # produced these labels. The QLoRA lane's hygiene rule admits only rows labeled by
        # the hardened validator -- without this stamp that filter needs timestamp forensics.
        "label_env": _label_env(),
    }
    return _write_row(row)


def log_selection(
    prompt: str,
    *,
    provenance: str,
    spec: dict[str, Any] | None = None,
    checks: dict[str, str] | None = None,
    claimable: bool | None = None,
    material: str | None = None,
    object_name: str | None = None,
    source: str = "live",
) -> bool:
    """Append a method-SELECTION row for a PHYSICS-authored part (analytic/topopt): no
    build123d code (the physics authors emit geometry directly), but the label the near-term
    flywheel actually learns -- WHICH method a typed spec routed to and whether its uniform
    gate was claimable. Same additive JSONL schema as ``log_part`` (``code`` is None), so one
    reader handles both. Fail-open."""
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "provenance": provenance,
        "prompt": prompt,
        "object_name": object_name,
        "material": material,
        "code": None,  # physics authors produce geometry, not code -- a SELECTION row
        "code_sha": None,
        "accepted": bool(claimable),
        "geometry_passed": None,
        "hard_ok": None,
        "critic_approved": None,
        "stopped_reason": None,
        "iterations": None,
        "spec": spec,
        "checks": checks,
        "claimable": claimable,
        "label_env": _label_env(),
    }
    return _write_row(row)


def load_training_rows(
    path: Path | None = None, *, hardened_only: bool = True
) -> list[dict[str, Any]]:
    """Read JSONL rows, excluding unstamped or stale labels by default.

    The append-only log contains historical rows from weaker validators. Training
    callers must opt out explicitly to inspect them; the default admits only rows
    stamped by the current validator fingerprint, feature-presence gate, strict final
    disposition, and no critic-refutation override.
    """
    source = path or _path()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    current = None
    if hardened_only:
        from forge_mind.validation_provenance import validation_fingerprint

        current = validation_fingerprint()
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if hardened_only:
            env = row.get("label_env")
            if not isinstance(env, dict):
                continue
            if env.get("validation_fingerprint") != current:
                continue
            if env.get("feature_presence") is not True:
                continue
            if env.get("strict_disposition") is not True:
                continue
            if env.get("critic_refutation_override") is True:
                continue
        rows.append(row)
    return rows


__all__ = ["load_training_rows", "log_part", "log_selection"]
