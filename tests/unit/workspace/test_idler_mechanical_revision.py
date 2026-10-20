"""The saved engineering job must survive independent STEP checks and detect defects."""

import json
from pathlib import Path

import build123d as b
from scripts.idler_mechanical_check import annulus, mechanical_checks
from scripts.idler_mechanical_revision import revised

from forge_mind.workspace.assembly import named_solids, occurrence_label
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document


def test_revised_export_and_adversarial_mechanical_defects(tmp_path):
    source = (
        Path(__file__).resolve().parents[3]
        / "docs/benchmarks/engineering-idler/source-revision.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    document = revised(DesignDocument.model_validate(raw["document"]))
    shape, _ = compile_document(document, {})
    step = tmp_path / "idler.step"
    b.export_step(shape, step)
    result = mechanical_checks(step)
    assert len(result["checks"]) == 45
    assert all(check["status"] == "pass" for check in result["checks"])
    assert all(check["status"] == "pass" for check in result["tool_access"])
    assert all(check["status"] == "pass" for check in result["rotation_bounds"])
    assert all(check["sampled_clear"] for check in result["disassembly"])
    assert result["status"] == "needs_review"  # no certificate from nominal geometry

    # Change the exported solids, leaving author features/labels untouched.
    solids = named_solids(b.import_step(step))
    base = occurrence_label("base_fixed")
    carrier = occurrence_label("moving/carrier")
    nut = occurrence_label("adjuster_retention")
    solids[base] = solids[base].fuse(b.Pos(-50, -30, 1) * b.Cylinder(3.4, 1))
    solids[base] = solids[base] - b.Pos(34, 12, 5) * b.Cylinder(3.1, 6)
    solids[carrier] = solids[carrier] - b.Pos(0, 31.5, 18) * b.Box(20, 3, 15)
    # Nut gap remains acceptable0.35, but shortening the shoulder adds0.05
    # hidden play. Measuring only the nut gap would incorrectly pass.
    solids[nut] = b.Pos(0.1, 0, 0) * solids[nut]
    adjuster = occurrence_label("adjuster")
    solids[adjuster] = solids[adjuster] - annulus(7, 4.01, 0.1, (30, 0, 18), "x")
    crossbolt = occurrence_label("hub_crossbolt")
    solids[crossbolt] = solids[crossbolt] - b.Pos(56, 2, 18) * b.Box(10, 14, 10)
    children = []
    for label, solid in solids.items():
        solid.label = label
        children.append(solid)
    broken = tmp_path / "broken.step"
    b.export_step(b.Compound(children=children), broken)
    failed = mechanical_checks(broken)
    failures = {c["id"] for c in failed["checks"] if c["status"] == "fail"}
    assert {
        "mount_through_-50_-30",
        "guide_wall_3mm_24",
        "screw_endplay",
        "support_thread_host_34_12",
        "crosspin_transverse_shank",
    } <= failures
    assert failed["status"] == "fail"
