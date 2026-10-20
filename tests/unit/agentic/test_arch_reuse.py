"""FORGE_REUSE_ARCH -- the resume half FORGE_REUSE_PLAN lacks (the L102/H5B incident):
plan reuse skips the fresh-arch stage, so a MECH_ARCH run resumed by plan alone has no
HardwareDecls and its `<servo>.horn` endpoints all read unknown-part -> no_build before
any checkpoint reloads. `_reload_mech_arch` reloads the saved typed doc, re-derives the
hardware exactly as the fresh stage does (0 LLM calls), and fail-opens archless."""

from __future__ import annotations

from pathlib import Path

from forge_mind.agentic.functional import _reload_mech_arch
from forge_mind.agentic.mech_arch import ActuatorPlacement, MechArchitecture, MechJoint


def _arch() -> MechArchitecture:
    return MechArchitecture(
        object_name="test rig",
        joints=[
            MechJoint(
                name="shoulder_L",
                kind="actuated",
                axis_dir="+X",
                station_mm=(0.0, 0.0, 100.0),
                bearing="Ø6.0 pin journal in Ø6.35 bore land, running fit 0.35",
                retention="printed shoulder + push-on domed cap",
            ),
            MechJoint(
                name="elbow_L",
                kind="passive",
                axis_dir="+X",
                station_mm=(0.0, 0.0, 50.0),
                bearing="Ø5.0 pin journal in Ø5.35 bore land, running fit 0.35",
                retention="printed shoulder + push-on cap",
            ),
        ],
        actuators=[
            ActuatorPlacement(
                joint="shoulder_L",
                hw_id="servo_shoulder_L",
                position_mm=(0.0, -20.0, 100.0),
                shaft_dir="+X",
                horn_link="upper_arm_L",
            )
        ],
        fastening_system="M3 self-tappers into 4.0 bosses everywhere",
        seam_strategy="tongue-and-groove lips, 4 screw bosses per seam",
        wiring_route="each actuator bay routes along the spine channel to the pelvis bus",
    )


def test_reload_derives_hardware_and_writes_out_copy(tmp_path: Path) -> None:
    src = tmp_path / "backup_mech_arch.json"
    src.write_text(_arch().model_dump_json(indent=2), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    arch_obj, arch_block, hardware = _reload_mech_arch(str(src), out, None)
    assert arch_obj is not None and arch_obj.object_name == "test rig"
    assert arch_block  # the decompose callout is rebuilt
    assert hardware and getattr(hardware[0], "hw_id", None) == "servo_shoulder_L"
    assert (out / "mech_arch.json").exists()  # postmortem copy rides the out dir


def test_reload_never_overwrites_caller_hardware(tmp_path: Path) -> None:
    src = tmp_path / "backup_mech_arch.json"
    src.write_text(_arch().model_dump_json(indent=2), encoding="utf-8")
    sentinel = [object()]
    _arch_obj, _block, hardware = _reload_mech_arch(str(src), tmp_path, sentinel)
    assert hardware is sentinel  # caller-passed decls win, same as the fresh stage


def test_corrupt_file_fail_opens_archless(tmp_path: Path) -> None:
    bad = tmp_path / "mech_arch.json.bak"
    bad.write_text("{not json", encoding="utf-8")
    arch_obj, arch_block, hardware = _reload_mech_arch(str(bad), tmp_path, None)
    assert arch_obj is None and arch_block is None and hardware is None
