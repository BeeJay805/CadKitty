"""PhysicsReport composer (ADR-0063) — proven money-free against REAL meshes + the real validator.

These tests encode the council's non-negotiable invariants: fail-closed not_run, the verdict is a PURE
MAP of the validator's aggregate (never re-aggregated from findings), the illustrative FEA never carries a
measured value, and a not_run finding never reads as `verified`.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.physics.fea import LoadResponse
from forge_mind.physics.report.compose import compose_physics_report
from forge_mind.physics.report.holes import DetectedHole, classify_hole
from forge_mind.physics.report.mates import evaluate_mate
from forge_mind.physics.report.projectors import (
    project_boltbrain,
    project_fitproof,
    project_loadproof,
)
from forge_mind.physics.report.schema import Finding, Intent, Mate, PhysicsReport
from forge_mind.types import Vec3
from forge_mind.validator.intrinsic import IntrinsicReport, validate_intrinsic
from forge_mind.validator.report import Check, CheckStatus, StepProof, Tier


def _stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    data = mesh.export(file_type="stl")
    return data if isinstance(data, bytes) else bytes(data, "utf-8")


def _compose_for(mesh: trimesh.Trimesh, tmp_path: Path) -> PhysicsReport:
    stl = tmp_path / "part.stl"
    stl.write_bytes(_stl_bytes(mesh))
    report = validate_intrinsic(stl)
    ext = mesh.extents
    return compose_physics_report(
        report,
        intent=Intent(),
        bbox_mm=Vec3(x=float(ext[0]), y=float(ext[1]), z=float(ext[2])),
        surface_area_mm2=float(mesh.area),
    )


def _good_box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(20.0, 20.0, 20.0))


def _two_body() -> trimesh.Trimesh:
    a = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    b = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    b.apply_translation((100.0, 0.0, 0.0))  # far apart -> two disconnected bodies
    return trimesh.util.concatenate([a, b])


def _find(report: PhysicsReport, finding_id: str) -> Finding | None:
    for s in report.suites:
        for f in s.findings:
            if f.id == finding_id:
                return f
    return None


def _all_findings(report: PhysicsReport) -> list[Finding]:
    return [f for s in report.suites for f in s.findings]


# ── verdict + mutation ──────────────────────────────────────────────────────────────────────────


def test_good_box_is_printable(tmp_path: Path) -> None:
    pr = _compose_for(_good_box(), tmp_path)
    assert pr.overall.verdict in {"pass", "warn"}


def test_blocking_warnings_are_scoped_to_the_printability_verdict(tmp_path: Path) -> None:
    # audit M5: the verdict is the PRINTABILITY aggregate; blocking/warnings are paired with it and
    # must count ONLY the printability suite. Counting across all suites let a non-printability
    # finding (fit/load) push blocking>=1 while verdict=="pass" -- a self-contradictory header.
    mesh = _good_box()
    stl = tmp_path / "part.stl"
    stl.write_bytes(_stl_bytes(mesh))
    report = validate_intrinsic(stl)
    ext = mesh.extents
    pr = compose_physics_report(
        report,
        intent=Intent(mates_with=[Mate(kind="press_fit", spec="20mm H7/p6 shaft")]),
        bbox_mm=Vec3(x=float(ext[0]), y=float(ext[1]), z=float(ext[2])),
        surface_area_mm2=float(mesh.area),
    )
    by_id = {s.id: s for s in pr.suites}
    printproof = by_id["printproof"]
    assert pr.overall.blocking == sum(1 for f in printproof.findings if f.status == "fail")
    assert pr.overall.warnings == sum(1 for f in printproof.findings if f.status == "warn")
    # a printable box stays "pass" even though the fit suite cannot be satisfied (no holes)
    if pr.overall.verdict == "pass":
        assert pr.overall.blocking == 0
    wt = _find(pr, "printproof.mesh.watertight")
    assert wt is not None and wt.status == "pass"
    # vitals computed (20 mm cube = 8000 mm^3, ~9.9 g PLA), CoM present and in the Z-up frame.
    assert pr.vitals.volume_mm3 is not None and 7900 < pr.vitals.volume_mm3 < 8100
    assert pr.vitals.mass_g is not None and pr.vitals.mass_g.material == "PLA"
    assert pr.vitals.center_of_mass_mm is not None
    assert pr.vitals.surface_area_mm2 is not None


def test_mutation_two_body_fails_the_report(tmp_path: Path) -> None:
    """The deliberately-bad part MUST fail a check AND flip the overall verdict — no silent green."""
    pr = _compose_for(_two_body(), tmp_path)
    sb = _find(pr, "printproof.mesh.single_body")
    assert sb is not None and sb.status == "fail" and sb.severity == "blocking"
    # measured is structured + honest (parsed from the check's own string), not a formatted blob.
    assert sb.measured is not None and sb.measured.value == 2 and sb.measured.unit == "bodies"
    assert pr.overall.verdict == "fail"
    assert pr.overall.blocking >= 1


def test_populated_vs_empty_differ(tmp_path: Path) -> None:
    g = tmp_path / "g"
    b = tmp_path / "b"
    g.mkdir()
    b.mkdir()
    good = _compose_for(_good_box(), g)
    bad = _compose_for(_two_body(), b)
    assert good.overall.verdict != bad.overall.verdict


def test_verdict_is_a_pure_map_never_recomputed_from_findings() -> None:
    """ADR-0056: even if every projected finding is PASS, a FAIL aggregate yields a FAIL verdict. The
    composer relabels the validator's verdict, it never re-rolls one from the findings."""
    rep = IntrinsicReport(
        printable=CheckStatus.FAIL,  # validator says fail
        verdict="Printable: NO.",
        checks=[
            Check(  # ...but the only check projected is a PASS
                id="mesh.watertight",
                tier=Tier.MESH,
                status=CheckStatus.PASS,
                message="ok",
                expected="watertight",
                actual="watertight",
            )
        ],
        metrics=None,
        stability=None,
    )
    pr = compose_physics_report(rep, intent=Intent(), bbox_mm=Vec3(), surface_area_mm2=None)
    assert pr.overall.verdict == "fail"  # follows `printable`, NOT the all-pass finding


# ── fail-closed + honesty ───────────────────────────────────────────────────────────────────────


def test_fea_is_not_run_without_a_load(tmp_path: Path) -> None:
    pr = _compose_for(_good_box(), tmp_path)
    fea = _find(pr, "loadproof.fea")
    assert fea is not None
    assert fea.status == "not_run"
    assert (
        fea.measured is None and fea.limit is None
    )  # illustrative FEA never fills a measured slot
    assert fea.severity == "advisory"  # cannot move overall.blocking
    assert fea.confidence.tier != "verified"  # honest tier on an unrun check
    assert fea.reason_not_run


def test_every_not_run_finding_has_an_honest_tier_and_reason(tmp_path: Path) -> None:
    """Founder ruling: a not_run finding never carries `verified`, and always says why."""
    pr = _compose_for(_good_box(), tmp_path)
    not_run = [f for f in _all_findings(pr) if f.status == "not_run"]
    assert not_run, "expected fit/bolt/fea stubs to be not_run"
    for f in not_run:
        assert f.confidence.tier != "verified", f"{f.id} claims verified while not_run"
        assert f.reason_not_run, f"{f.id} is not_run with no reason"
        assert f.measured is None and f.limit is None


def test_fit_and_bolt_are_not_run_stubs(tmp_path: Path) -> None:
    pr = _compose_for(_good_box(), tmp_path)
    fit = _find(pr, "fitproof.intent")
    bolt = _find(pr, "boltbrain.holes")
    assert fit is not None and fit.status == "not_run" and "mat" in fit.reason_not_run.lower()
    assert bolt is not None and bolt.status == "not_run" and "step" in bolt.reason_not_run.lower()


def test_stability_is_a_real_advisory_finding(tmp_path: Path) -> None:
    pr = _compose_for(_good_box(), tmp_path)
    stab = _find(pr, "loadproof.stability")
    assert stab is not None
    assert stab.status in {"pass", "warn"}  # the box has a flat base -> it runs
    assert stab.confidence.tier == "computed"
    assert stab.severity == "advisory"  # reported-only, can never gate the verdict
    assert stab.locator is not None and stab.locator.point is not None


# ── contract shape ──────────────────────────────────────────────────────────────────────────────


def test_four_suites_no_domainproof(tmp_path: Path) -> None:
    pr = _compose_for(_good_box(), tmp_path)
    ids = [s.id for s in pr.suites]
    assert ids == ["printproof", "fitproof", "boltbrain", "loadproof"]
    assert "domainproof" not in ids


def test_serializes_to_the_frozen_contract_shape(tmp_path: Path) -> None:
    pr = _compose_for(_two_body(), tmp_path)
    data = pr.model_dump(mode="json", by_alias=True)
    assert data["schema_version"] == 2  # v2: the optional ADR-0086 motion block
    assert data["scope"] == "single_part"
    # no verdict, summary, or explanation ever says "safe" (ADR-0056 honesty ceiling)
    blob = str(data).lower()
    assert "safe" not in blob
    # the mandatory not-a-guarantee disclaimer is present and non-removable
    assert any("not a guarantee" in d for d in data["disclaimers"])
    # any fix.change serializes the python-keyword key as `from`, not `from_`
    for s in data["suites"]:
        for f in s["findings"]:
            if f.get("fix") and f["fix"].get("change"):
                assert "from" in f["fix"]["change"] and "from_" not in f["fix"]["change"]


def test_min_wall_limit_matches_what_actually_ran(tmp_path: Path) -> None:
    """The limit is parsed from the check's own `expected`, so it always equals the floor that ran
    (0.8 mm intrinsic), never a hardcoded fixture value (2.0 mm)."""
    pr = _compose_for(_good_box(), tmp_path)
    mw = _find(pr, "printproof.features.min_wall_thickness")
    assert mw is not None and mw.limit is not None
    assert mw.limit.value == 0.8 and mw.limit.comparator == ">="


def test_blocking_count_never_contradicts_the_verdict(tmp_path: Path) -> None:
    """A 'fail' verdict must report >= 1 blocking issue (counts are status-driven), so the header can
    never read 'fail, 0 blocking'. A failing part fails via a fail-status finding even when that check's
    severity badge is 'warning' (e.g. an overhang)."""
    bad = _compose_for(_two_body(), tmp_path)
    assert bad.overall.verdict == "fail"
    assert bad.overall.blocking >= 1
    assert bad.overall.blocking == sum(
        1 for s in bad.suites for f in s.findings if f.status == "fail"
    )


# ── slice 2: STEP integrity ───────────────────────────────────────────────────────────────────────


def _intrinsic_with_step(step_proof: StepProof) -> IntrinsicReport:
    return IntrinsicReport(
        printable=CheckStatus.PASS,
        verdict="ok",
        checks=[
            Check(
                id="mesh.watertight",
                tier=Tier.MESH,
                status=CheckStatus.PASS,
                message="ok",
                expected="watertight",
                actual="watertight",
            )
        ],
        metrics=None,
        stability=None,
        step_proof=step_proof,
    )


def _brep_step_proof() -> StepProof:
    return StepProof(
        matches=True,
        volume_mm3=8000.0,
        area_mm2=2400.0,
        centroid=Vec3(),
        volume_rel_err=0.0,
        area_rel_err=0.0,
        centroid_dist_mm=0.0,
        schema_id="AP214",
        is_brep=True,
        tessellated=False,
        product_name="bracket",
        named=True,
        solid_count=1,
        face_count=6,
        topology_matches=True,
    )


def test_step_brep_emits_an_editable_cad_finding() -> None:
    pr = compose_physics_report(
        _intrinsic_with_step(_brep_step_proof()),
        intent=Intent(),
        bbox_mm=Vec3(),
        surface_area_mm2=None,
    )
    si = _find(pr, "printproof.step_integrity")
    assert si is not None
    assert si.status == "pass" and si.severity == "advisory"
    assert (
        si.measured is not None and si.measured.value == 6 and si.measured.unit == "analytic faces"
    )
    assert si.confidence.tier == "verified"
    # BoltBrain stays not_run, but no longer says "needs a STEP" (the hole geometry is present now).
    bolt = _find(pr, "boltbrain.holes")
    assert bolt is not None and bolt.status == "not_run"
    assert bolt.reason_not_run is not None and "needs a step" not in bolt.reason_not_run.lower()


def test_step_tessellated_lump_warns_with_a_fix() -> None:
    sp = _brep_step_proof().model_copy(
        update={"is_brep": False, "tessellated": True, "matches": False}
    )
    pr = compose_physics_report(
        _intrinsic_with_step(sp), intent=Intent(), bbox_mm=Vec3(), surface_area_mm2=None
    )
    si = _find(pr, "printproof.step_integrity")
    assert si is not None and si.status == "warn"
    assert si.fix is not None and "B-rep" in si.fix.text


def test_stl_upload_has_no_step_integrity_finding(tmp_path: Path) -> None:
    pr = _compose_for(_good_box(), tmp_path)  # an STL -> report.step_proof is None
    assert _find(pr, "printproof.step_integrity") is None


# ── slice 3: hole detection + BoltBrain classification ────────────────────────────────────────────


def _hole(
    diameter_mm: float,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    wall_mm: float | None = None,
) -> DetectedHole:
    return DetectedHole(
        diameter_mm=diameter_mm,
        center_mm=Vec3(x=x, y=y, z=z),
        axis=Vec3(z=1.0),
        depth_mm=5.0,
        wall_mm=wall_mm,
    )


def test_classify_hole_matches_iso_sizes() -> None:
    m3 = classify_hole(3.4)  # ISO 273 M3 normal clearance
    assert m3 is not None and m3.size == "M3" and m3.kind == "clearance"
    m5 = classify_hole(5.5)  # M5 normal clearance
    assert m5 is not None and m5.size == "M5"
    tap = classify_hole(2.5)  # M3 coarse tap drill
    assert tap is not None and tap.size == "M3" and tap.kind == "tap drill"
    assert classify_hole(9.5) is None  # between M8 sizes -> non-standard, not snapped


def test_boltbrain_classifies_and_groups_holes() -> None:
    holes = [_hole(3.4, x=10), _hole(3.4, x=20), _hole(5.5, x=30)]
    fs = project_boltbrain(holes, has_brep=True)
    m3 = next((f for f in fs if "m3" in f.id), None)
    m5 = next((f for f in fs if "m5" in f.id), None)
    assert m3 is not None and m5 is not None
    assert m3.status == "info" and m3.severity == "advisory"
    assert m3.confidence.tier == "verified"
    assert (
        m3.locator is not None and m3.locator.point is not None
    )  # located on a representative hole
    assert m3.measured is not None and m3.measured.value == 3.4
    assert "2 holes" in m3.explanation  # the two M3 holes grouped


def test_boltbrain_nonstandard_hole_gets_a_fix() -> None:
    fs = project_boltbrain([_hole(9.5)], has_brep=True)
    nf = next((f for f in fs if "nonstandard" in f.id), None)
    assert nf is not None and nf.fix is not None and nf.locator is not None


def test_boltbrain_no_holes_on_step_is_not_run() -> None:
    fs = project_boltbrain([], has_brep=True)
    assert len(fs) == 1 and fs[0].status == "not_run"
    assert fs[0].reason_not_run is not None and "detected" in fs[0].reason_not_run.lower()


def test_boltbrain_stl_no_holes_needs_step() -> None:
    fs = project_boltbrain([], has_brep=False)
    assert fs[0].status == "not_run"
    assert fs[0].reason_not_run is not None and "step" in fs[0].reason_not_run.lower()


def test_compose_threads_holes_into_boltbrain() -> None:
    pr = compose_physics_report(
        _intrinsic_with_step(_brep_step_proof()),
        intent=Intent(),
        bbox_mm=Vec3(),
        surface_area_mm2=None,
        holes=[_hole(3.4, x=5, wall_mm=6.0)],
    )
    bolt = [f for s in pr.suites if s.id == "boltbrain" for f in s.findings]
    m3 = next((f for f in bolt if "m3" in f.id), None)
    assert m3 is not None and m3.status == "info"  # holes present -> real ID finding, not the stub


# ── slice 4: edge distance + small-hole printability (intent-free) ────────────────────────────────


def test_edge_distance_warns_when_hole_too_close() -> None:
    # M3 (Ø3.4) with only 0.5 mm of wall -> ratio 0.5 + 0.5/3.4 = 0.65 < 1.5 -> warn + fix, located.
    fs = project_boltbrain([_hole(3.4, wall_mm=0.5)], has_brep=True)
    ed = next(f for f in fs if f.id == "boltbrain.edge_distance")
    assert ed.status == "warn" and ed.severity == "warning"
    assert ed.fix is not None and ed.locator is not None and ed.locator.point is not None


def test_edge_distance_passes_when_clear() -> None:
    fs = project_boltbrain([_hole(3.4, wall_mm=6.0)], has_brep=True)  # ratio 2.26 >= 1.5
    ed = next(f for f in fs if f.id == "boltbrain.edge_distance")
    assert ed.status == "pass" and ed.fix is None


def test_edge_distance_not_run_when_unmeasured() -> None:
    fs = project_boltbrain([_hole(3.4, wall_mm=None)], has_brep=True)
    ed = next(f for f in fs if f.id == "boltbrain.edge_distance")
    assert ed.status == "not_run" and ed.reason_not_run  # fail closed, never a silent pass


def test_min_hole_warns_for_tiny_bore() -> None:
    fs = project_boltbrain([_hole(1.5, wall_mm=5.0)], has_brep=True)  # Ø1.5 < 2.0 mm floor
    mh = next((f for f in fs if f.id == "boltbrain.min_hole"), None)
    assert mh is not None and mh.status == "warn" and mh.locator is not None


def test_no_min_hole_finding_for_normal_bore() -> None:
    fs = project_boltbrain([_hole(3.4, wall_mm=5.0)], has_brep=True)
    assert not any(f.id == "boltbrain.min_hole" for f in fs)


# ── slice 5: FitProof — bore-vs-mate (ISO 273 bolts / ISO 286 shafts / bearing seat) ──────────────


def test_fit_bolt_clearance_pass_and_fail() -> None:
    ok = evaluate_mate("bolt", "M5", [_hole(5.5)])  # ISO 273 M5 normal clearance
    assert ok is not None and ok.status == "pass" and ok.label.startswith("M5")
    tight = evaluate_mate(
        "bolt", "M5", [_hole(5.0)]
    )  # below M5 close clearance (5.3) -> won't pass
    assert tight is not None and tight.status == "fail"


def test_fit_shaft_iso286() -> None:
    ok = evaluate_mate("shaft", "8H7/h6", [_hole(8.02)])  # bore clears the h6 shaft -> running fit
    assert ok is not None and ok.status == "pass"
    interfere = evaluate_mate("shaft", "8H7/h6", [_hole(7.9)])  # bore < shaft max -> interference
    assert interfere is not None and interfere.status == "fail"


def test_fit_bearing_seat() -> None:
    ok = evaluate_mate("bearing", "608", [_hole(22.0)])  # 608 OD = 22 mm
    assert ok is not None and ok.status == "pass"
    loose = evaluate_mate("bearing", "608", [_hole(22.5)])  # 0.5 mm over OD -> spins
    assert loose is not None and loose.status == "fail"


def test_fit_unparseable_or_unknown_returns_none() -> None:
    assert evaluate_mate("bolt", "banana", [_hole(5.5)]) is None  # bad spec
    assert evaluate_mate("widget", "M5", [_hole(5.5)]) is None  # unknown kind -> not_run upstream


def test_project_fitproof_gating() -> None:
    assert project_fitproof([], [_hole(5.5)])[0].status == "not_run"  # no mate
    assert project_fitproof([Mate(kind="bolt", spec="M5")], [])[0].status == "not_run"  # no holes


def test_project_fitproof_located_verified_finding() -> None:
    fs = project_fitproof([Mate(kind="bolt", spec="M5")], [_hole(5.5, x=3.0)])
    f = fs[0]
    assert f.suite == "fitproof" and f.status == "pass"
    assert f.confidence.tier == "verified"
    assert f.locator is not None and f.locator.point is not None
    assert f.limit is not None and "ISO 273" in f.limit.source


def test_compose_fitproof_unlocked_by_intent() -> None:
    pr = compose_physics_report(
        _intrinsic_with_step(_brep_step_proof()),
        intent=Intent(mates_with=[Mate(kind="bolt", spec="M5")]),
        bbox_mm=Vec3(),
        surface_area_mm2=None,
        holes=[_hole(5.5, x=3.0)],
    )
    fit = [f for s in pr.suites if s.id == "fitproof" for f in s.findings]
    assert any(
        f.status == "pass" for f in fit
    )  # intent + matching hole -> a real decision-grade fit


# ── slice 8: LoadProof real-load verdict (decision-grade FEA) ──────────────────────────────────────


def _load_resp() -> LoadResponse:
    return LoadResponse(
        max_deflection_mm=4.2,
        deflection_at_mm=(21.0, 38.0, 13.0),
        max_von_mises_mpa=18.5,
        stress_at_mm=(6.0, 21.0, 4.0),
        load_n=49.0,
        load_case="49 N over the top face, base fixed, direction (0.00, 0.00, -1.00)",
        material_e_mpa=3500.0,
        note="Isotropic FEA; an FDM part is weaker across layers — not a safety guarantee.",
    )


def test_loadproof_with_response_emits_located_deflection_and_stress() -> None:
    fs = project_loadproof(None, _load_resp(), load_requested=True)
    ids = {f.id for f in fs}
    assert "loadproof.deflection" in ids and "loadproof.stress" in ids
    defl = next(f for f in fs if f.id == "loadproof.deflection")
    assert defl.status == "info" and defl.confidence.tier == "computed"
    assert defl.measured is not None and defl.measured.value == 4.2 and defl.measured.unit == "mm"
    assert defl.locator is not None and defl.locator.point is not None
    stress = next(f for f in fs if f.id == "loadproof.stress")
    assert stress.measured is not None and stress.measured.unit == "MPa"
    assert (
        "FDM" in stress.explanation or "layers" in stress.explanation
    )  # the derate caveat rides along


def test_loadproof_load_requested_but_no_response_is_not_run() -> None:
    fs = project_loadproof(None, None, load_requested=True)
    fea = next(f for f in fs if f.id == "loadproof.fea")
    assert fea.status == "not_run"
    assert fea.reason_not_run is not None and "did not run" in fea.reason_not_run.lower()


def test_loadproof_no_load_is_not_run() -> None:
    fs = project_loadproof(None, None, load_requested=False)
    fea = next(f for f in fs if f.id == "loadproof.fea")
    assert fea.status == "not_run"
    assert fea.reason_not_run is not None and "no load" in fea.reason_not_run.lower()


def test_compose_loadproof_unlocked_by_load() -> None:
    pr = compose_physics_report(
        _intrinsic_with_step(_brep_step_proof()),
        intent=Intent(load={"magnitude": 5, "unit": "kg"}),  # type: ignore[arg-type]
        bbox_mm=Vec3(),
        surface_area_mm2=None,
        load_response=_load_resp(),
    )
    lp = [f for s in pr.suites if s.id == "loadproof" for f in s.findings]
    assert any(f.id == "loadproof.deflection" for f in lp)
