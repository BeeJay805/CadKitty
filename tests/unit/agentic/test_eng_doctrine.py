"""Engineering levers (founder 2026-07-23 evening: H3 is "detailed but not engineered
-- no real company would put this out"). Every interface must carry a NAMED engineering
decision a deterministic audit can verify: actuator housing, horn coupling, pin
retention. FORGE_ENG_DOCTRINE (prompt overlay) and FORGE_ENG_AUDIT (plan gate) default
OFF, byte-identical off; FORGE_EXPERT_REPLAN (blocker-gated re-plan) defaults OFF.

The audit tests pin the REAL archived H3 card strings (the incident that motivated the
lever), per the lane law: regressions pin to measured strings, not synthetic
approximations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge_mind.agentic.functional import (
    _DECOMPOSE_SYSTEM,
    _ENG_DOCTRINE,
    _FIDELITY_DOCTRINE,
    _decompose_system,
    _plan_engineering_issues,
    _plan_form_issues,
    _plan_mirror_issues,
)
from forge_mind.structures.servo_bridge import HardwareDecl

# The ACTUAL H3 shoulder-pin card (benchmarks/2026-07-23_humanoid_probe/H3/plan.json):
# contains the word "shoulder" (bore) yet retains NOTHING -- the exact string that
# forces multi-word retention tokens (bare "shoulder" would false-pass it).
_H3_SHOULDER_PIN = (
    "Model the LEFT shoulder pin of a 350 mm consumer humanoid in the shared frame (Z up "
    "from sole z=0, +Y robot's left): a plain Ø6.0 mm x 40 mm cylindrical steel-style pin, "
    "axis along the Y direction at height z=290, x=0, spanning y=45 to y=85, with 0.5 mm "
    "chamfers on both ends. Its inboard section (y=45 to 56) seats in the torso's Ø6.2 "
    "shoulder bore; its outboard section (y=61 to 85) is the journal the upper arm's Ø6.3 "
    "bore rotates on (0.3 mm diametral running clearance)."
)
# The H2 torso card's servo-window phrasing (the honest pocket H3 dropped): states the
# body dims AND a pocket word, so the housing check must PASS it.
_H2_STYLE_TORSO = (
    "Front plate: two THROUGH rectangular servo windows, each 41.2 mm x 20.2 mm, sized "
    "for an MG996R body 40.7 x 19.7 mm plus 0.25 mm air per side, with the 4-hole lug "
    "pattern on 49.5 x 10.0."
)


def _plan(*parts: tuple[str, str]):
    return SimpleNamespace(
        parts=[SimpleNamespace(name=n, prompt=p, retention=None) for n, p in parts]
    )


def _servos(n: int = 2) -> list[HardwareDecl]:
    return [HardwareDecl(kind="mg996r_servo", hw_id=f"servo_{i}") for i in range(n)]


def test_eng_doctrine_names_the_decision_classes() -> None:
    assert "ENGINEERING DOCTRINE" in _ENG_DOCTRINE
    assert "NAMED ENGINEERING DECISION" in _ENG_DOCTRINE
    assert "EVERY MOVING JOINT IS A BEARING PROBLEM" in _ENG_DOCTRINE
    assert "BOUGHT ACTUATORS ARE GEOMETRY" in _ENG_DOCTRINE
    assert "SHELL-AND-RIB, NOT SOLID" in _ENG_DOCTRINE
    assert "MIRRORED PAIRS ARE MIRRORS" in _ENG_DOCTRINE
    # The clearance CLASSES (per function, never one number everywhere).
    for cls in ("0.30-0.40", "0.15-0.20", "0.05"):
        assert cls in _ENG_DOCTRINE
    # The datasheet lug pattern the housing card must carry.
    assert "49.5 x 10.0" in _ENG_DOCTRINE


def test_decompose_system_eng_flag_composes_and_off_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for flag in (
        "FORGE_PLAN_DOCTRINE",
        "FORGE_FIDELITY_DOCTRINE",
        "FORGE_ENG_DOCTRINE",
        "FORGE_CHIEF",
    ):
        monkeypatch.delenv(flag, raising=False)
    assert _decompose_system() == _DECOMPOSE_SYSTEM
    monkeypatch.setenv("FORGE_ENG_DOCTRINE", "1")
    assert _decompose_system() == _DECOMPOSE_SYSTEM + _ENG_DOCTRINE
    monkeypatch.setenv("FORGE_FIDELITY_DOCTRINE", "1")
    assert _decompose_system() == _DECOMPOSE_SYSTEM + _FIDELITY_DOCTRINE + _ENG_DOCTRINE


def test_h3_shoulder_pin_fires_retention_despite_the_word_shoulder() -> None:
    # The measured collision case: "shoulder bore" contains "shoulder", and the pin
    # retains nothing -- the audit must still fire (multi-word tokens earn their keep).
    issues = _plan_engineering_issues(_plan(("shoulder_pin_L", _H3_SHOULDER_PIN)), None)
    assert len(issues) == 1 and "shoulder_pin_L" in issues[0]
    assert "retention" in issues[0]


def test_retained_pin_is_clean() -> None:
    retained = _H3_SHOULDER_PIN + (
        " Outboard end carries an e-clip groove 0.9 mm wide at y=83 (axial retention)."
    )
    assert _plan_engineering_issues(_plan(("shoulder_pin_L", retained)), None) == []


def test_declared_servos_demand_housing_and_horn() -> None:
    # H3's situation reconstructed: servos declared, no card states the body pocket,
    # no card mentions a horn -- both engineering checks fire.
    plan = _plan(("chest_plate", "a crowned breastplate shell, 2.4 mm wall"))
    issues = _plan_engineering_issues(plan, _servos(2))
    assert sum("housing card" in i for i in issues) == 2  # one per declared servo
    assert sum("HORN" in i for i in issues) == 1


def test_h2_style_window_card_passes_housing_and_horn_coupling_clears() -> None:
    plan = _plan(
        ("torso", _H2_STYLE_TORSO),
        ("upper_arm_L", "the arm root carries the servo horn bore circle (6 holes, 14.0)"),
    )
    assert _plan_engineering_issues(plan, _servos(2)) == []


def test_no_hardware_is_vacuous_on_actuator_checks() -> None:
    # No declared hardware -> housing/horn checks are vacuous (nothing to house);
    # a pinless plan has no retention duty either. Off = silent.
    plan = _plan(("bracket", "a plain bracket, two M4 counterbores"))
    assert _plan_engineering_issues(plan, None) == []


# ---- L101: retention is TYPED first; the token grep is the legacy fallback ---------------

_H5 = __import__("pathlib").Path(__file__).resolve().parents[2] / "fixtures" / "h5"


def _h5_pair():
    from forge_mind.agentic.functional import FunctionalPlan, _canonicalize_plan_endpoints
    from forge_mind.agentic.mech_arch import MechArchitecture

    plan = FunctionalPlan.model_validate_json((_H5 / "plan.json").read_text(encoding="utf-8"))
    arch = MechArchitecture.model_validate_json(
        (_H5 / "mech_arch.json").read_text(encoding="utf-8")
    )
    return _canonicalize_plan_endpoints(plan), arch


def test_h5_false_fail_correction_quotes_the_arch_decision() -> None:
    """The REAL H5 pair (verbatim copy; the run's retention was fully DESIGNED --
    shoulder flanges modeled, press caps declared as parts -- and the token grep
    still refused it). Until the field is filled the audit fires, but the correction
    must now quote the architecture's own decision through the Joint.name carrier."""
    plan, arch = _h5_pair()
    issues = [i for i in _plan_engineering_issues(plan, None, arch) if "retention" in i]
    pins = {"elbow_pin_L", "elbow_pin_R", "knee_pin_L", "knee_pin_R"}
    assert {p for p in pins if any(f"'{p}'" in i for i in issues)} == pins
    # every correction carries the copyable decision, not a token list
    assert all("push-on domed cap" in i and "`retention` field" in i for i in issues)


def test_h5_with_the_field_filled_is_clean() -> None:
    """The plan the corrected decomposer emits: retention copied verbatim from the
    arch through the named joints -> the audit is satisfied by the FIELD."""
    plan, arch = _h5_pair()
    ret_by_joint = {j.name: j.retention for j in arch.joints}
    by_pin: dict[str, str] = {}
    for j in plan.joints:
        if j.name in ret_by_joint:
            for r in (j.a, j.b):
                part = r.split(".", 1)[0]
                if "pin" in part:
                    by_pin[part] = ret_by_joint[j.name]
    filled = plan.model_copy(
        update={
            "parts": [
                p.model_copy(update={"retention": by_pin[p.name]}) if p.name in by_pin else p
                for p in plan.parts
            ]
        }
    )
    issues = [i for i in _plan_engineering_issues(filled, None, arch) if "retention" in i]
    assert issues == []


def test_card_prompt_renders_the_retention_line_only_when_set() -> None:
    from forge_mind.agentic.functional import PartPlan, _card_prompt

    bare = PartPlan(name="elbow_pin_L", prompt="a stepped pin")
    assert _card_prompt(bare) == "a stepped pin"  # legacy checkpoints keep their keys
    typed = PartPlan(
        name="elbow_pin_L",
        prompt="a stepped pin",
        retention="printed shoulder + push-on domed cap",
    )
    out = _card_prompt(typed)
    assert "AXIAL RETENTION (BINDING" in out and "push-on domed cap" in out
    assert out.startswith("a stepped pin")


# ---- mirror-sibling completeness (the probe-P2 false-positive fix) -----------------
# The REAL P2 roster (benchmarks/2026-07-23_humanoid_probe/probes/P2/plan.json): every
# L/R part has its sibling present, so the corrected check must stay SILENT -- the old
# probe check fired 6 times here only because it compared against arch.mirrored_pairs.
_P2_ROSTER = [
    "head",
    "torso",
    "pelvis",
    "upper_arm_L",
    "upper_arm_R",
    "forearm_L",
    "forearm_R",
    "elbow_pin_L",
    "elbow_pin_R",
    "thigh_L",
    "thigh_R",
    "shin_foot_L",
    "shin_foot_R",
    "knee_pin_L",
    "knee_pin_R",
]


def test_mirror_silent_when_every_lr_part_has_its_sibling() -> None:
    plan = _plan(*((n, f"card for {n}") for n in _P2_ROSTER))
    assert _plan_mirror_issues(plan) == []


def test_mirror_fires_on_a_real_roster_asymmetry() -> None:
    # The guard against weakening: drop shin_foot_R -> the lone shin_foot_L must fire.
    roster = [n for n in _P2_ROSTER if n != "shin_foot_R"]
    issues = _plan_mirror_issues(_plan(*((n, "card") for n in roster)))
    assert len(issues) == 1 and "shin_foot_L" in issues[0] and "asymmetry" in issues[0]


def test_mirror_handles_long_left_right_suffixes() -> None:
    plan = _plan(("foot_left", "card"), ("foot_right", "card"), ("wrist_left", "card"))
    issues = _plan_mirror_issues(plan)
    assert len(issues) == 1 and "wrist_left" in issues[0]


# ---- box-as-final-form (the probe-P2 pelvis false-positive fix) --------------------
# The ACTUAL P2 pelvis sentence: an engineered rounded loft that EXPLICITLY forbids a box.
_P2_PELVIS = (
    "Central shell: rounded-rectangular loft 58 wide (X) x 48 deep (Y), z=152..195, "
    "corner R10, wall 2.5, gently crowned faces (no bare box);"
)


def test_form_passes_an_engineered_no_bare_box_card() -> None:
    assert _plan_form_issues(_plan(("pelvis", _P2_PELVIS))) == []


def test_form_fires_on_a_real_box_limb() -> None:
    issues = _plan_form_issues(_plan(("thigh_L", "the thigh is a box 34 x 30 x 75, filleted")))
    assert len(issues) == 1 and "thigh_L" in issues[0]


def test_form_ignores_measurement_and_structural_box_terms() -> None:
    plan = _plan(
        ("frame", "sized to its bounding box of 40 x 40 x 12"),
        ("beam", "a box-section spar 20 x 20, wall 2"),
    )
    assert _plan_form_issues(plan) == []
