"""The hardware= wire-up (ADR-0124): pins the R3 design's one load-bearing gap.

The measured defect (docs/research/2026-07-22-r3-harness-design.md): the servo bridge
(`placed_hardware_for`) and the adapter's ``hardware=`` parameter both existed and were
tested, but `_validate_motion` never passed hardware through -- so a registry-mapped
servo was invisible to the mount-pattern and envelope-sweep gates on the live path.
These tests pin the threading contract itself: decls are resolved via the bridge and
arrive at ``from_part_results(hardware=...)``."""

from __future__ import annotations

from types import SimpleNamespace

import forge_mind.agentic.functional as functional
from forge_mind.agentic.functional import _validate_motion


def _plan_with_one_joint() -> SimpleNamespace:
    parts = [
        SimpleNamespace(name="base", ports=[SimpleNamespace(name="p1")]),
        SimpleNamespace(name="arm", ports=[SimpleNamespace(name="p2")]),
    ]
    joints = [SimpleNamespace(type="revolute", a="base.p1", b="arm.p2")]
    return SimpleNamespace(parts=parts, joints=joints)


def _capture_adapter(captured: dict):
    def fake_from_part_results(part_results, joints, ports_by_part, notes=None, *, hardware=None):
        captured["hardware"] = hardware
        # Return a model that fails the >=2-bodies check so _validate_motion exits early
        # on the "unvalidated" path -- the threading is what this test measures.
        return SimpleNamespace(bodies=[], joints=[])

    return fake_from_part_results


def test_declared_hardware_reaches_the_adapter(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "forge_mind.agentic.assembly_adapter.from_part_results", _capture_adapter(captured)
    )
    resolved = SimpleNamespace(tag="placed-servo")
    monkeypatch.setattr(
        "forge_mind.structures.servo_bridge.placed_hardware_for",
        lambda decl, *, workdir=None: resolved,
    )
    decl = SimpleNamespace(kind="mg996r_servo", hw_id="pan_servo")
    _validate_motion(
        [("base", SimpleNamespace()), ("arm", SimpleNamespace())],
        _plan_with_one_joint(),
        hardware=[decl],
    )
    assert captured["hardware"] == [resolved]


def test_no_hardware_is_byte_identical_none(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "forge_mind.agentic.assembly_adapter.from_part_results", _capture_adapter(captured)
    )
    _validate_motion(
        [("base", SimpleNamespace()), ("arm", SimpleNamespace())], _plan_with_one_joint()
    )
    assert captured["hardware"] is None


def test_unresolvable_hardware_degrades_with_a_note(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "forge_mind.agentic.assembly_adapter.from_part_results", _capture_adapter(captured)
    )

    def boom(decl, *, workdir=None):
        raise KeyError("unknown hardware kind")

    monkeypatch.setattr("forge_mind.structures.servo_bridge.placed_hardware_for", boom)
    decl = SimpleNamespace(kind="nope", hw_id="ghost")
    verdict, _model, _report = _validate_motion(
        [("base", SimpleNamespace()), ("arm", SimpleNamespace())],
        _plan_with_one_joint(),
        hardware=[decl],
    )
    # Never raises; the failed decl resolves to an empty placed list, and the run
    # carries a drop note naming the ghost (verdict is the unvalidated dict here
    # because the fake adapter returns a degenerate model).
    assert captured["hardware"] == []
    assert verdict is not None and any("ghost" in n for n in verdict.get("joint_drop_notes", []))


def test_topology_accepts_declared_hardware_endpoints() -> None:
    # The attempt-3 incident: `pan_servo.horn` refused as "a part the plan does not
    # declare". With the hardware port map, servo endpoints are legal; unknown ports
    # and undeclared hardware still gate.
    from forge_mind.agentic.functional import _plan_topology_issues

    plan = SimpleNamespace(
        parts=[SimpleNamespace(name="bracket", ports=[SimpleNamespace(name="hub")])],
        joints=[SimpleNamespace(type="revolute", a="pan_servo.horn", b="bracket.hub")],
    )
    hw = {"pan_servo": {"horn", "mount"}}
    assert _plan_topology_issues(plan, hw) == []
    # unknown port on declared hardware gates
    bad_port = SimpleNamespace(
        parts=plan.parts,
        joints=[SimpleNamespace(type="revolute", a="pan_servo.spline", b="bracket.hub")],
    )
    assert any("exposes ports" in i for i in _plan_topology_issues(bad_port, hw))
    # undeclared hardware still gates
    assert any(
        "does not declare" in i for i in _plan_topology_issues(plan, {"other_servo": {"horn"}})
    )


def test_hardware_mate_kills_the_floating_false_fail() -> None:
    # The L81 incident: the cradle mated ONLY the servo and read 'floating'. With the
    # hardware mesh in the scan it seats; hardware never mints interference verdicts.
    import trimesh

    from forge_mind.agentic.functional import _assembly_fit

    a = trimesh.creation.box(extents=(10, 10, 10))
    b = trimesh.creation.box(extents=(10, 10, 10))
    b.apply_translation((11.0, 0, 0))  # 1mm gap to a -- touching (snug) per the check
    lone = trimesh.creation.box(extents=(8, 8, 8))
    lone.apply_translation((100.0, 0, 0))  # far from every printed part
    servo = trimesh.creation.box(extents=(8, 8, 8))
    servo.apply_translation((100.0, 0, 7.5))  # overlaps `lone` -- its hardware mount

    without = _assembly_fit([a, b, lone])
    assert without is not None and 2 in without["floating_parts"]

    with_hw = _assembly_fit([a, b, lone], hardware_meshes=[servo])
    assert with_hw is not None
    assert with_hw["floating_parts"] == []
    assert with_hw["hardware_mated_parts"] == [2]
    # deep hardware overlap must NOT create an interference verdict (sweep's business)
    assert with_hw["seated"] is True


# Keep a reference so the import is exercised even if functional's internals move.
_ = functional
