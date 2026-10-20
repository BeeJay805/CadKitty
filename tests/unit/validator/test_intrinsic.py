"""validate_intrinsic (ADR-0056 Flavor A) — the intent-free pre-print check.

The load-bearing guarantees: no false PASS (a broken or unbuildable mesh must NOT come back printable),
the report can never be mistaken for a fit/function certificate (it omits the intent tiers and stamps
fit_function_verified=False), and an unreadable model fails closed.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.validator.intrinsic import IntrinsicReport, validate_intrinsic
from forge_mind.validator.report import CheckStatus, Tier


def _export(mesh: trimesh.Trimesh, tmp_path: Path, name: str) -> Path:
    stl = tmp_path / f"{name}.stl"
    mesh.export(str(stl))
    return stl


def _ids(report: IntrinsicReport, status: CheckStatus) -> set[str]:
    return {c.id for c in report.checks if c.status == status}


def test_clean_solid_box_is_printable_but_not_fit_verified(tmp_path: Path) -> None:
    stl = _export(trimesh.creation.box(extents=(20.0, 20.0, 20.0)), tmp_path, "box")
    report = validate_intrinsic(stl)

    assert report.printable != CheckStatus.FAIL
    assert "mesh.watertight" in _ids(report, CheckStatus.PASS)
    # honest ceiling: intrinsic NEVER claims fit/function.
    assert report.fit_function_verified is False
    assert "NOT verified" in report.verdict
    assert "safe" not in report.verdict.lower()
    # REPORTED physics ride along (mass/CoM + stand-or-tip).
    assert report.metrics is not None and report.metrics.volume_mm3 > 0
    assert report.stability is not None and report.stability.stable is True


def test_non_watertight_mesh_is_not_printable(tmp_path: Path) -> None:
    # A box with a face removed has a boundary hole — not a solid. Must NOT come back printable.
    box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    open_mesh = trimesh.Trimesh(vertices=box.vertices.copy(), faces=box.faces[:-2].copy())
    stl = _export(open_mesh, tmp_path, "open")
    report = validate_intrinsic(stl)

    assert report.printable == CheckStatus.FAIL
    assert "mesh.watertight" in _ids(report, CheckStatus.FAIL)
    assert "Printable: NO" in report.verdict


def test_degenerate_thin_sheet_is_not_printable(tmp_path: Path) -> None:
    # Smallest dimension below one printable layer (0.1 mm) — physically unbuildable.
    stl = _export(trimesh.creation.box(extents=(50.0, 50.0, 0.05)), tmp_path, "sheet")
    report = validate_intrinsic(stl)

    assert report.printable == CheckStatus.FAIL
    assert "mesh.min_printable_thickness" in _ids(report, CheckStatus.FAIL)


def test_intrinsic_omits_the_intent_tiers(tmp_path: Path) -> None:
    # Without declared intent there is no interface / hardware-conformance / assembly check — so the
    # report can't silently look like a full validate() PASS.
    stl = _export(trimesh.creation.box(extents=(20.0, 20.0, 20.0)), tmp_path, "box")
    report = validate_intrinsic(stl)

    tiers = {c.tier for c in report.checks}
    assert Tier.INTERFACES not in tiers
    assert Tier.ASSEMBLY not in tiers
    assert tiers <= {Tier.MESH, Tier.FEATURES, Tier.PRINTABILITY}
    assert report.fit_function_verified is False


def test_unreadable_model_fails_closed(tmp_path: Path) -> None:
    report = validate_intrinsic(tmp_path / "does_not_exist.stl")
    assert report.printable == CheckStatus.FAIL
