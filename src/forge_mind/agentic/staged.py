"""Staged construction: build the part the way a CAD feature tree does, verifying each node.

The complex-parts arc's central move (plan: complex-parts-arc.md; premortem PM-1..PM-7).
The monolithic loop authors the WHOLE part per iteration, so feedback is diffuse and every
rewrite risks features that already worked (row 50: the duct's 8-iteration thrash). Staged
construction replaces ONLY the initial draft construction: base body first, then one stage
per brief feature, each stage sandbox-built and checkpoint-verified before the next. The
result then enters the UNCHANGED standard machinery (validate -> critic -> iterate ->
param-opt -> judge) -- staged is a smarter draft constructor, not a second pipeline.

Laws (from the premortem):
- FORGE_STAGED_BUILD default OFF; flag-off is byte-identical (PM-1).
- ONE PARAMS dict across stages (param-opt / stated-dim pinning / attribution key off it) (PM-2).
- Checkpoint checks are SCOPED to what an intermediate can honestly satisfy: sandbox-ok,
  watertight, envelope upper bound, hole-census non-regression, volume band, and the stage's
  own features measured. Full min-wall/DFM gates run only on the FINAL part (PM-3).
- Prefix-lock is ADVISORY (recorded, never gates): the MEASURED non-regression check is the
  authority on "you broke prior geometry" (PM-7).
- A stage that cannot converge in its tries STOPS staging and hands the last good partial to
  the standard loop, whose missing-feature machinery completes it -- staging can slow a part
  down but never hard-fail it.
- Every stage writes a record to <work_dir>/stages.json: the fine-grained data exhaust
  (tier-6 seed / future fine-tune trajectories).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from forge_mind.agentic.author import author_initial, author_revision
from forge_mind.agentic.brief import DesignBrief, parse_envelope_mm
from forge_mind.agentic.params import extract_params
from forge_mind.agentic.sandbox import run_build123d_code
from forge_mind.fabrication import FabricationMode

_STAGE_TRIES = 2
_ENVELOPE_SLACK = 1.25  # intermediates may not exceed the brief envelope by more than this
_VOLUME_FLOOR = 0.35  # one stage may not silently erase more than 65% of the prior solid
_DIA_BIN_MM = 0.5  # hole-census diameter binning


def staged_enabled() -> bool:
    return os.environ.get("FORGE_STAGED_BUILD", "").strip().lower() in {"1", "true", "on"}


def staged_min_features() -> int:
    """Below this feature count the overhead cannot pay (PM-4); the monolithic path runs."""
    try:
        return max(1, int(os.environ.get("FORGE_STAGED_MIN_FEATURES", "4")))
    except ValueError:
        return 4


@dataclass
class StageRecord:
    """One stage's verdict trail -- persisted join-ready in stages.json."""

    name: str
    tries: int = 0
    built: bool = False
    refused: bool = False
    prefix_kept: bool | None = None  # advisory (PM-7)
    feature_verified: str = "unchecked"  # "measured" | "weak-volume" | "unchecked"
    issue: str | None = None
    volume_mm3: float | None = None
    census: dict[str, int] = field(default_factory=dict)


@dataclass
class StagedResult:
    code: str | None  # the accumulated program (last GOOD stage; None if stage 0 never built)
    approach: str
    cost_usd: float
    records: list[StageRecord]
    completed: int
    total_stages: int


def _mesh_stats(stl_path: Path | None) -> tuple[float | None, bool | None, Any]:
    if stl_path is None or not stl_path.exists():
        return None, None, None
    try:
        import trimesh

        mesh: Any = trimesh.load(str(stl_path), force="mesh")
        return float(mesh.volume), bool(mesh.is_watertight), mesh.bounds
    except Exception:
        return None, None, None


def _hole_census(step_path: Path | None) -> dict[str, int]:
    """Concave-cylinder counts binned by diameter -- the non-regression fingerprint."""
    if step_path is None or not step_path.exists():
        return {}
    try:
        from build123d import import_step

        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        part = import_step(str(step_path))
        out: dict[str, int] = {}
        for s in cluster_coaxial(extract_cylinders(part)):
            if s.concave is not True:
                continue
            key = f"{round((2.0 * float(s.radius)) / _DIA_BIN_MM) * _DIA_BIN_MM:.1f}"
            out[key] = out.get(key, 0) + 1
        return out
    except Exception:
        return {}  # unmeasurable -> the census check silently narrows, never blocks


def _census_regressions(prior: dict[str, int], current: dict[str, int]) -> list[str]:
    out = []
    for d, n in prior.items():
        if current.get(d, 0) < n:
            out.append(f"holes at ~{d}mm dropped {n} -> {current.get(d, 0)}")
    return out


def _excise_params(code: str) -> str:
    """The program with its PARAMS literal removed -- the advisory prefix-lock text."""
    spec = extract_params(code)
    if spec is None:
        return code
    return code[: spec.start] + code[spec.end :]


def _prefix_kept(prior: str, new: str) -> bool:
    """Advisory (PM-7): did every non-PARAMS line of the prior program survive, in order?
    Real additions land INSIDE the build block (before ``result = part.part``), so this is
    ordered line containment, not literal prefixing."""

    def _lines(code: str) -> list[str]:
        return [ln.strip() for ln in _excise_params(code).splitlines() if ln.strip()]

    want = _lines(prior)
    have = iter(_lines(new))
    return all(any(w == h for h in have) for w in want)


_HOLE_FEATURE_TOKENS = frozenset(
    {"hole", "holes", "bore", "bores", "port", "ports", "dowel", "clearance", "drill", "tapped"}
)


def _feature_expectation(feat: Any) -> tuple[float | None, int]:
    """(hole diameter, count) -- ONLY when the feature reads hole-like by NAME/PURPOSE.

    First-contact bug (interaction_v0 A/B, dense_gearbox_face staged k0): matching on the
    DIMS text alone made a "main disc body -- 110 mm diameter x 8 mm" stage demand a 110mm
    HOLE. Discs, bosses, necks and barrels all have diameters; only hole-like features may
    set a census expectation -- everything else takes the weak-volume path."""
    import re

    words = set(
        re.findall(
            r"[a-z]+",
            (str(getattr(feat, "name", "")) + " " + str(getattr(feat, "purpose", ""))).lower(),
        )
    )
    if not (words & _HOLE_FEATURE_TOKENS):
        return None, int(getattr(feat, "count", None) or 1)
    dims = str(getattr(feat, "approx_dims_mm", "") or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm\s*)?(?:dia(?:meter)?|[øØ⌀])", dims) or re.search(
        r"[øØ⌀]\s*(\d+(?:\.\d+)?)", dims
    )
    count = int(getattr(feat, "count", None) or 1)
    return (float(m.group(1)) if m else None), count


_BODY_HEADS = frozenset(
    {
        "body",
        "shell",
        "disc",
        "plate",
        "base",
        "frame",
        "housing",
        "block",
        "stock",
        "blank",
        "wall",
        "walls",
    }
)


def _is_body_feature(feat: Any) -> bool:
    """Body-class features (shells/discs/plates) are authored WITH the base massing --
    the A/B measured that giving them their own stage yields a no-op or a coplanar
    re-fuse (row 53). Head-final naming, same convention as spatial's hole plan."""
    import re

    words = re.findall(r"[a-z]+", str(getattr(feat, "name", "")).lower())
    return bool(words) and words[-1] in _BODY_HEADS


def _stage_feedback(feat: Any, attempt_issue: str | None) -> str:
    dims = str(getattr(feat, "approx_dims_mm", "") or "")
    purpose = str(getattr(feat, "purpose", "") or "")
    fb = (
        "STAGED CONSTRUCTION -- ADD ONE FEATURE ONLY.\n"
        f"The program so far is correct and LOCKED. Append the code that adds exactly this "
        f"feature and nothing else:\n  feature: {feat.name}\n  purpose: {purpose}\n"
        f"  dims: {dims}\n"
        "Rules: do NOT modify or reorder existing geometry code; keep the ONE existing PARAMS "
        "dict and ADD any new keys to it (never a second dict); return the FULL program with "
        "your addition appended."
    )
    if attempt_issue:
        fb += f"\n\nYOUR PREVIOUS ATTEMPT FAILED ITS CHECKPOINT: {attempt_issue}. Fix exactly that."
    return fb


def staged_initial(
    brief: DesignBrief,
    *,
    client: Any = None,
    model: str | None = None,
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    work_dir: Path,
    sandbox_limits: Any = None,
    part_name: str = "part",
) -> StagedResult:
    """Base body first, then one verified stage per brief feature. Returns the accumulated
    program of the last GOOD stage (partial on stage refusal -- the standard loop finishes)."""
    records: list[StageRecord] = []
    cost = 0.0
    body_feats = [f for f in brief.features if _is_body_feature(f)]
    stage_feats = [f for f in brief.features if not _is_body_feature(f)]
    total_stages = 1 + len(stage_feats)
    env = parse_envelope_mm(brief.overall_envelope_mm)

    def _checkpoint(
        rec: StageRecord,
        stl: Path | None,
        step: Path | None,
        prior_census: dict[str, int],
        prior_vol: float | None,
    ) -> str | None:
        vol, watertight, bounds = _mesh_stats(stl)
        rec.volume_mm3 = vol
        if watertight is not True:
            return "the intermediate solid is not watertight"
        if env is not None and bounds is not None:
            built = sorted(float(bounds[1][i] - bounds[0][i]) for i in range(3))
            for got, cap in zip(built, sorted(env), strict=True):
                if got > cap * _ENVELOPE_SLACK:
                    return (
                        f"intermediate already exceeds the brief envelope: {got:.1f}mm vs "
                        f"cap {cap:.0f}mm (x{_ENVELOPE_SLACK} slack)"
                    )
        census = _hole_census(step)
        rec.census = census
        regress = _census_regressions(prior_census, census)
        if regress:
            return "prior features regressed: " + "; ".join(regress)
        if prior_vol is not None and vol is not None and vol < prior_vol * _VOLUME_FLOOR:
            return (
                f"the stage erased most of the prior solid (volume {prior_vol:.0f} -> "
                f"{vol:.0f} mm^3)"
            )
        return None

    # ---- stage 0: the base body + every BODY-CLASS feature (the A/B's measured lesson:
    # the base massing already builds shells/discs/plates, so giving them their own stages
    # produces either a no-op ("changed nothing") or a coplanar re-fuse that breaks
    # watertightness -- staging then bails to a bare base and pays its cost without its
    # benefit (row 53, dense_junction_box + dense_pump_manifold_flange). Body-class
    # features are AUTHORED WITH the base; only non-body features get stages. ------------
    base_label = "__base__" + (
        f" (+{', '.join(str(f.name) for f in body_feats)[:80]})" if body_feats else ""
    )
    base_rec = StageRecord(name=base_label)
    # The base KEEPS target_mass_g. It authors the body-class features -- the shells, discs and
    # plates that ARE the bulk -- so it is the only call that can size the part to a mass; later
    # stages add or (usually) subtract comparatively small features. Third review-catch on this
    # line: clearing it here left NO call able to act on the target, because `_stage_feedback`
    # tells each stage to add its one feature and not resize existing geometry. Middle stages
    # still get it cleared (see the loop below) so none of them chases the whole volume alone.
    # Honest bound: for a staged build the target is approximate -- subtractive stages shave the
    # base after it was sized -- so mass fidelity is best on the monolithic path.
    base_brief = brief.model_copy(update={"features": body_feats})
    code: str | None = None
    approach = ""
    prior_census: dict[str, int] = {}
    prior_vol: float | None = None
    issue: str | None = None
    for _attempt in range(_STAGE_TRIES):
        base_rec.tries += 1
        try:
            if code is None:
                authored, usage = author_initial(
                    base_brief, client=client, model=model, fabrication_mode=fabrication_mode
                )
            else:
                authored, usage = author_revision(
                    base_brief,
                    code,
                    f"the base body failed its checkpoint: {issue}. Fix exactly that.",
                    client=client,
                    model=model,
                    fabrication_mode=fabrication_mode,
                )
        except Exception as exc:
            issue = f"author error: {type(exc).__name__}"
            continue
        cost += usage.cost_usd
        code, approach = authored.code, authored.approach
        sres = run_build123d_code(
            code, work_dir / "stage_0", part_name=part_name, limits=sandbox_limits
        )
        if not sres.ok:
            issue = f"build failed: {(sres.error_message or '')[:200]}"
            continue
        issue = _checkpoint(base_rec, sres.stl_path, sres.step_path, {}, None)
        if issue is None:
            base_rec.built = True
            prior_census, prior_vol = dict(base_rec.census), base_rec.volume_mm3
            break
    if not base_rec.built:
        base_rec.refused, base_rec.issue = True, issue
        records.append(base_rec)
        _persist(work_dir, records)
        return StagedResult(None, approach, cost, records, 0, total_stages)
    records.append(base_rec)

    # ---- one stage per feature -------------------------------------------------------------
    assert code is not None  # the base stage built, or we returned above
    good_code: str = code
    completed = 1
    for ix, feat in enumerate(stage_feats, start=1):
        rec = StageRecord(name=str(feat.name))
        issue = None
        # MIDDLE stages never chase the whole-part volume (review-caught): one told to reach the
        # finished solid while adding a single feature oversizes itself, and every later stage
        # piles on. The FINAL stage keeps it so the last call can true the part up against the
        # target it is about to ship (the base already sized the bulk against it).
        stage_brief = (
            brief if ix == len(stage_feats) else brief.model_copy(update={"target_mass_g": None})
        )
        for _attempt in range(_STAGE_TRIES):
            rec.tries += 1
            try:
                authored, usage = author_revision(
                    stage_brief,
                    good_code,
                    _stage_feedback(feat, issue),
                    client=client,
                    model=model,
                    fabrication_mode=fabrication_mode,
                )
            except Exception as exc:
                issue = f"author error: {type(exc).__name__}"
                continue
            cost += usage.cost_usd
            candidate = authored.code
            rec.prefix_kept = _prefix_kept(good_code, candidate)  # advisory only (PM-7)
            sres = run_build123d_code(
                candidate, work_dir / f"stage_{ix}", part_name=part_name, limits=sandbox_limits
            )
            if not sres.ok:
                issue = f"build failed: {(sres.error_message or '')[:200]}"
                continue
            issue = _checkpoint(rec, sres.stl_path, sres.step_path, prior_census, prior_vol)
            if issue is not None:
                continue
            # the stage's OWN feature: measured for hole features, volume-delta otherwise
            want_d, want_n = _feature_expectation(feat)
            if want_d is not None:
                key = f"{round(want_d / _DIA_BIN_MM) * _DIA_BIN_MM:.1f}"
                gained = rec.census.get(key, 0) - prior_census.get(key, 0)
                if gained < want_n:
                    issue = (
                        f"the stage was supposed to add {want_n} hole(s) of ~{want_d}mm but the "
                        f"measured census gained {gained}"
                    )
                    continue
                rec.feature_verified = "measured"
            else:
                if (
                    prior_vol is not None
                    and rec.volume_mm3 is not None
                    and abs(rec.volume_mm3 - prior_vol) < max(1.0, prior_vol * 0.0005)
                ):
                    issue = "the stage changed nothing measurable (volume unchanged)"
                    continue
                rec.feature_verified = "weak-volume"
            rec.built = True
            good_code = candidate
            prior_census, prior_vol = dict(rec.census), rec.volume_mm3
            completed += 1
            break
        if not rec.built:
            rec.refused, rec.issue = True, issue
            records.append(rec)
            break  # hand the partial to the standard loop (never hard-fail)
        records.append(rec)

    _persist(work_dir, records)
    return StagedResult(good_code, approach, cost, records, completed, total_stages)


def _persist(work_dir: Path, records: list[StageRecord]) -> None:
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "stages.json").write_text(
            json.dumps([asdict(r) for r in records], indent=1), encoding="utf-8"
        )
    except Exception:
        pass  # the record is exhaust, never a failure source


__all__ = ["StageRecord", "StagedResult", "staged_enabled", "staged_initial", "staged_min_features"]
