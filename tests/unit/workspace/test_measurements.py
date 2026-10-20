from __future__ import annotations

from pathlib import Path

import pytest
from build123d import Box, Cylinder, Pos, export_step, export_stl

from forge_mind.workspace.measure import measure_revision, section_revision, verify_artifacts
from forge_mind.workspace.models import Contract, Requirement


def export(directory: Path, width: float = 40, diameter: float = 6, count: int = 1) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    part = Box(width, 30, 8)
    if count:
        part -= Cylinder(diameter / 2, 12)
    if count > 1:
        part -= Pos(12, 0, 0) * Cylinder(diameter / 2, 12)
    export_step(part, directory / "part.step")
    export_stl(part, directory / "part.stl")
    (directory / "program.py").write_text("# independent test geometry", encoding="utf-8")


def contract() -> Contract:
    return Contract(
        requirements=[
            Requirement(
                id="width",
                text="40 mm wide",
                source_quote="40 mm",
                source="explicit",
                metric="extent_x",
                value=40,
            ),
            Requirement(
                id="bore",
                text="6 mm bore at origin",
                source_quote="6 mm",
                source="explicit",
                metric="hole_diameter",
                value=6,
                center=(0, 0),
            ),
            Requirement(
                id="count",
                text="one bore",
                source_quote="one",
                source="explicit",
                metric="hole_count",
                value=1,
            ),
        ]
    )


def test_known_geometry_is_measured_from_step(tmp_path: Path) -> None:
    export(tmp_path)
    report = measure_revision(tmp_path, "r1", contract())
    assert report.geometry_valid, report.errors
    assert report.requirements_satisfied, report.findings
    assert report.facts["extents_mm"] == pytest.approx([40, 30, 8])
    assert verify_artifacts(tmp_path, report)


@pytest.mark.parametrize(
    "width,diameter,count,failing",
    [
        (41, 6, 1, "width"),
        (40, 5, 1, "bore"),
        (40, 6, 0, "count"),
        (40, 6, 2, "count"),
    ],
)
def test_independent_mutations_do_not_earn_approval(
    tmp_path: Path, width: float, diameter: float, count: int, failing: str
) -> None:
    export(tmp_path, width, diameter, count)
    report = measure_revision(tmp_path, "r1", contract())
    assert not report.requirements_satisfied
    assert next(f for f in report.findings if f.requirement_id == failing).status != "pass"


def test_ambiguous_hole_is_not_assigned_by_desired_diameter(tmp_path: Path) -> None:
    export(tmp_path, count=2)
    c = contract()
    c.requirements[1].center = None
    report = measure_revision(tmp_path, "r1", c)
    assert report.findings[1].status == "unverified"


def test_unknown_and_empty_contracts_never_certify(tmp_path: Path) -> None:
    export(tmp_path)
    c = Contract(
        requirements=[Requirement(id="load", text="holds 5 kg", source_quote="holds 5 kg")]
    )
    assert not measure_revision(tmp_path, "r", c).requirements_satisfied
    assert not measure_revision(tmp_path, "r", Contract()).requirements_satisfied


def test_changed_or_missing_artifact_invalidates_evidence(tmp_path: Path) -> None:
    export(tmp_path)
    measured = measure_revision(tmp_path, "r", contract())
    (tmp_path / "program.py").write_text("different", encoding="utf-8")
    assert not verify_artifacts(tmp_path, measured)


def test_step_and_mesh_disagreement_is_not_valid(tmp_path: Path) -> None:
    export(tmp_path)
    export_stl(Box(10, 10, 10), tmp_path / "part.stl")
    report = measure_revision(tmp_path, "r", contract())
    assert not report.geometry_valid


def test_equal_volume_different_shape_is_not_export_agreement(tmp_path: Path) -> None:
    export(tmp_path, count=0)
    export_stl(Box(20, 30, 16), tmp_path / "part.stl")
    report = measure_revision(tmp_path, "r", contract())
    assert not report.geometry_valid
    assert not report.requirements_satisfied


def test_half_cylinder_edge_cutout_cannot_certify_a_complete_bore(tmp_path: Path) -> None:
    export(tmp_path, count=0)
    part = Box(40, 30, 8) - Pos(20, 0, 0) * Cylinder(3, 12)
    export_step(part, tmp_path / "part.step")
    export_stl(part, tmp_path / "part.stl")
    c = contract()
    c.requirements[1].center = (20, 0)
    report = measure_revision(tmp_path, "r", c)
    assert report.geometry_valid
    assert report.findings[1].status == "unverified"
    assert report.findings[2].status == "unverified"


def test_section_has_a_real_opening(tmp_path: Path) -> None:
    export(tmp_path)
    result = section_revision(tmp_path, "z", 0, tmp_path / "section.png")
    assert result["loops"] == 2
    assert (tmp_path / "section.png").is_file()
