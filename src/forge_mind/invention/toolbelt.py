"""Toolbelt + preflight v0 (ADR-0048, P6) — known tools, not vague capabilities.

Before a run the lifecycle should know exactly which tools are available (handoff 25.5). This is a
small capability REGISTRY + a DETERMINISTIC preflight: each tool has capabilities, the roles that
need it, a permission level, and a fallback policy; the preflight reports each tool's status and an
overall verdict. A required tool that is missing BLOCKS (no hallucinated workaround).

With Fable (or any model) unavailable, the preflight is honest: the OFFLINE build mode is available
(deterministic + mocked-seam work), LIVE concept generation is BLOCKED, and NO fake fallback model
output is ever produced -- the report says so in plain language rather than pretending.
"""

from __future__ import annotations

import importlib.util
import os
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_FIXTURE = Path(__file__).parents[3] / "benchmarks" / "invention_v1" / "problems.json"


class ToolStatus(StrEnum):
    AVAILABLE = "available"
    NOT_CONFIGURED = (
        "not_configured"  # the seam exists but nothing is wired (honest, not a failure)
    )
    UNAVAILABLE = "unavailable"  # required and missing -> a blocker


class Permission(StrEnum):
    READ_ONLY = "read_only"
    WRITE_ARTIFACT = "write_artifact"
    MUTATE_STATE = "mutate_state"


class ToolRegistryEntry(BaseModel):
    """One tool the lifecycle may use: what it does, who needs it, what it may touch, its fallback,
    and whether the OFFLINE build requires it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_id: str
    capabilities: list[str]
    required_for_roles: list[str]
    permission: Permission
    fallback: str | None = None
    required_for_offline: bool = False


# The invention lifecycle's tools (reviewed in git).
REGISTRY: list[ToolRegistryEntry] = [
    ToolRegistryEntry(
        tool_id="cad_engine",
        capabilities=["produce_part", "compose_generated_part"],
        required_for_roles=["mechanical", "validator"],
        permission=Permission.WRITE_ARTIFACT,
        required_for_offline=True,
    ),
    ToolRegistryEntry(
        tool_id="validators",
        capabilities=["mesh", "features", "printability", "interfaces", "assembly"],
        required_for_roles=["validator"],
        permission=Permission.READ_ONLY,
        required_for_offline=True,
    ),
    ToolRegistryEntry(
        tool_id="artifact_storage",
        capabilities=["write_stl", "write_step", "write_glb", "write_report"],
        required_for_roles=["mechanical", "documentation"],
        permission=Permission.WRITE_ARTIFACT,
        required_for_offline=True,
    ),
    ToolRegistryEntry(
        tool_id="vendor_fetch",
        capabilities=["search_listings"],
        required_for_roles=["real_parts_researcher"],
        permission=Permission.READ_ONLY,
        fallback="NullFetcher (fail-closed: every requirement FLAGs, no fake parts)",
    ),
    ToolRegistryEntry(
        tool_id="research_provider",
        capabilities=["web_search_documents"],
        required_for_roles=["requirements_researcher"],
        permission=Permission.READ_ONLY,
        fallback="None (fail-closed: problem-mode research BLOCKED; facts route to ASK, never faked)",
    ),
    ToolRegistryEntry(
        tool_id="model_client",
        capabilities=["front_door_concepts", "support_answers"],
        required_for_roles=["president", "concept_inventor", "support_copilot"],
        permission=Permission.READ_ONLY,
        fallback="offline: injected mocks in tests; NO fake fallback output at runtime",
    ),
    ToolRegistryEntry(
        tool_id="benchmark_fixtures",
        capabilities=["invention_v1"],
        required_for_roles=["benchmark"],
        permission=Permission.READ_ONLY,
        required_for_offline=True,
    ),
]


class ToolCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_id: str
    status: ToolStatus
    detail: str
    required_for_offline: bool
    fallback: str | None = None


class PreflightReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checks: list[ToolCheck] = Field(default_factory=list)
    offline_build_available: bool
    live_concept_generation: Literal["available", "blocked"]
    blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        """The offline build can proceed (every offline-required tool is available)."""
        return self.offline_build_available and not self.blockers


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _model_client_status() -> tuple[ToolStatus, str]:
    """Honest model-client status from the env (mirrors ``llm.provider.make_client`` resolution).
    Never pings a model -- it reports configuration, so the LIVE run is where an actual outage
    surfaces (and fails clearly). No fake fallback is ever implied."""
    provider = os.environ.get("FORGE_LLM_PROVIDER", "").strip()
    forbid = os.environ.get("FORGE_FORBID_API_SPEND", "").strip() == "1"
    if provider == "claude_code":
        return ToolStatus.AVAILABLE, "subscription provider (claude_code) configured"
    if provider:
        return ToolStatus.UNAVAILABLE, f"unknown FORGE_LLM_PROVIDER {provider!r}"
    if forbid:
        return ToolStatus.UNAVAILABLE, "metered spend is banned and no subscription provider is set"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return ToolStatus.AVAILABLE, "metered API key configured"
    return ToolStatus.NOT_CONFIGURED, "no model client configured (offline build still works)"


def _status_for(entry: ToolRegistryEntry, *, fixtures_path: Path) -> tuple[ToolStatus, str]:
    if entry.tool_id == "cad_engine":
        ok = _module_available("forge_mind.agentic.result")
        return (
            (ToolStatus.AVAILABLE, "produce_part importable")
            if ok
            else (
                ToolStatus.UNAVAILABLE,
                "the agentic engine is not importable",
            )
        )
    if entry.tool_id == "validators":
        ok = _module_available("forge_mind.validator.run")
        return (
            (ToolStatus.AVAILABLE, "5-tier validator importable")
            if ok
            else (
                ToolStatus.UNAVAILABLE,
                "the validator is not importable",
            )
        )
    if entry.tool_id == "artifact_storage":
        return ToolStatus.AVAILABLE, "local filesystem (per-job out_dir)"
    if entry.tool_id == "vendor_fetch":
        return (
            ToolStatus.NOT_CONFIGURED,
            "no live catalog wired; NullFetcher is the fail-closed default",
        )
    if entry.tool_id == "model_client":
        return _model_client_status()
    if entry.tool_id == "research_provider":
        from forge_mind.invention.research import get_research_provider

        provider = get_research_provider()
        if provider is not None:
            return (
                ToolStatus.AVAILABLE,
                f"{type(provider).__name__} configured (FORGE_RESEARCH_PROVIDER)",
            )
        return (
            ToolStatus.NOT_CONFIGURED,
            "no research provider configured; problem-mode research is BLOCKED (facts route to ASK)",
        )
    if entry.tool_id == "benchmark_fixtures":
        return (
            (ToolStatus.AVAILABLE, f"{fixtures_path.name} present")
            if fixtures_path.exists()
            else (
                ToolStatus.UNAVAILABLE,
                f"{fixtures_path} is missing",
            )
        )
    return ToolStatus.NOT_CONFIGURED, "unknown tool"


def run_preflight(*, fixtures_path: Path | None = None) -> PreflightReport:
    """Run the deterministic preflight over the tool registry. A required-for-offline tool that is
    UNAVAILABLE becomes a blocker; the model client never blocks the offline build (mocks cover it)."""
    fixtures = fixtures_path or _FIXTURE
    checks: list[ToolCheck] = []
    blockers: list[str] = []
    for entry in REGISTRY:
        status, detail = _status_for(entry, fixtures_path=fixtures)
        checks.append(
            ToolCheck(
                tool_id=entry.tool_id,
                status=status,
                detail=detail,
                required_for_offline=entry.required_for_offline,
                fallback=entry.fallback,
            )
        )
        if entry.required_for_offline and status is ToolStatus.UNAVAILABLE:
            blockers.append(f"{entry.tool_id}: {detail}")

    model = next(c for c in checks if c.tool_id == "model_client")
    live: Literal["available", "blocked"] = (
        "available" if model.status is ToolStatus.AVAILABLE else "blocked"
    )
    notes = [
        "Offline build mode is AVAILABLE: deterministic + mocked-seam work needs no live model.",
        "Mocked / injected tests are available for every model seam.",
        "NO fake fallback model output is produced -- a live run fails clearly if the model is down.",
    ]
    if live == "blocked":
        notes.insert(1, "LIVE concept generation is BLOCKED until a model client is configured.")
    return PreflightReport(
        checks=checks,
        offline_build_available=not blockers,
        live_concept_generation=live,
        blockers=blockers,
        notes=notes,
    )


__all__ = [
    "REGISTRY",
    "Permission",
    "PreflightReport",
    "ToolCheck",
    "ToolRegistryEntry",
    "ToolStatus",
    "run_preflight",
]
