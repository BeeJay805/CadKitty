"""Joint-resolution hardening (row 58): the radial declared 12 joints and the adapter's
exact-string port matching dropped ALL of them silently. Ports now match
slug-insensitively (canonicalized onto the declared name), every drop is named in the
``notes`` sink, and the drop notes become chief-visible failure lines."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import trimesh

from forge_mind.agentic.assembly_adapter import from_part_results
from forge_mind.agentic.functional import _assembly_failure_lines
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3


def _lr(tmp_path: Path, name: str) -> SimpleNamespace:
    stl = tmp_path / f"{name}.stl"
    trimesh.creation.box(extents=(10, 10, 10)).export(str(stl))
    return SimpleNamespace(final_stl=stl, final_step=None)


def _port(name: str) -> Port:
    return Port(
        name=name,
        origin=Vec3(x=0.0, y=0.0, z=0.0),
        axis=Vec3(x=1.0, y=0.0, z=0.0),
        normal=Vec3(x=0.0, y=0.0, z=1.0),
    )


def test_prose_port_reference_resolves_to_slug_declared_port(tmp_path: Path) -> None:
    # the row-58 class: joint says 'main axis', the part declared 'main-axis'
    parts = [("crankshaft", _lr(tmp_path, "crank")), ("block", _lr(tmp_path, "block"))]
    ports = {"crankshaft": [_port("main-axis")], "block": [_port("main-axis")]}
    joints = [
        Joint(
            type=JointType.REVOLUTE,
            a="crankshaft.main axis",
            b="block.main axis",
            range_deg=(0.0, 360.0),
        )
    ]
    notes: list[str] = []
    model = from_part_results(parts, joints, ports, notes=notes)
    assert len(model.joints) == 1, notes
    assert model.joints[0].a.endswith(".main-axis")  # canonicalized
    assert notes == []


def test_unknown_port_is_dropped_with_a_named_reason(tmp_path: Path) -> None:
    parts = [("crankshaft", _lr(tmp_path, "crank")), ("block", _lr(tmp_path, "block"))]
    ports = {"crankshaft": [_port("main-axis")], "block": [_port("main-axis")]}
    joints = [
        Joint(
            type=JointType.REVOLUTE,
            a="crankshaft.rear-stub",
            b="block.main-axis",
            range_deg=(0.0, 360.0),
        )
    ]
    notes: list[str] = []
    model = from_part_results(parts, joints, ports, notes=notes)
    assert model.joints == []
    assert len(notes) == 1
    assert "rear-stub" in notes[0] and "main-axis" in notes[0]  # says what WAS declared


def test_drop_notes_become_chief_failure_lines() -> None:
    motion = {
        "articulates": False,
        "motion_detail": "2 of 12 declared joints could not be resolved: joint x: ...",
        "joint_drop_notes": [
            "joint crankshaft.main axis <-> block.bore: port 'main axis' not declared",
            "joint rod-3.big end <-> crankshaft.pin: port 'big end' not declared",
        ],
        "checks": {},
    }
    lines = _assembly_failure_lines(None, motion, ["crankshaft", "block", "rod-3"])
    joined = "\n".join(lines)
    assert "joint declaration:" in joined
    assert "'main axis' not declared" in joined
    assert "rod-3.big end" in joined


def test_drop_notes_survive_the_validated_path(tmp_path: Path) -> None:
    """The north-star smoke reported 9/13 joints validated with ZERO trace: drop notes
    were attached only to the UNVALIDATED verdict. On a built + validated model they must
    ride the verdict too (they also feed the chief's failure lines)."""
    from forge_mind.agentic.functional import FunctionalPlan, PartPlan, _validate_motion

    parts_lr = [("base", _lr(tmp_path, "base")), ("lever", _lr(tmp_path, "lever"))]
    plan = FunctionalPlan(
        object_name="test mechanism",
        parts=[
            PartPlan(name="base", prompt="base card", ports=[_port("pivot")]),
            PartPlan(name="lever", prompt="lever card", ports=[_port("pivot")]),
        ],
        assembly_note="lever pivots on base",
        joints=[
            Joint(type=JointType.REVOLUTE, a="base.pivot", b="lever.pivot", range_deg=(0.0, 90.0)),
            Joint(
                type=JointType.REVOLUTE, a="base.pivot", b="lever.missing", range_deg=(0.0, 90.0)
            ),
        ],
    )
    verdict, model, _report = _validate_motion(parts_lr, plan)  # type: ignore[arg-type]
    assert verdict is not None and model is not None
    assert verdict["joints_declared"] == 2 and verdict["joints_validated"] == 1
    assert any("missing" in n for n in verdict["joint_drop_notes"])
    # the COMPLETE report fail list rides the verdict (deep-review: checks were born
    # ungated behind a whitelist) -- this fixture's report carries at least the
    # no-rod-in-bore pin_coaxial FAIL, and every entry names its check id
    fails = verdict["assembly_fails"]
    assert fails and all("id" in f and "message" in f for f in fails)
    assert any(f["id"] == "assembly.pin_coaxial" for f in fails)
