"""Stated-dimension protection (ADR-0080): the protected-number set is anchored to the USER
(prompt/intake/registry, never the model's brief numbers); the acceptance audit is non-degrade
(only optimizer-ADDED drift rejects), differentially measurable (erasure rejects), and fails
CLOSED when it cannot run.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.spatial import _HoleSpec
from forge_mind.agentic.stated_dims import (
    _judge_bores,
    _judge_envelope,
    stated_dim_audit,
    stated_numbers,
)


def _brief(*features: BriefFeature, envelope: str = "40 x 40 x 10") -> DesignBrief:
    return DesignBrief(
        object_name="test part",
        summary="a test part",
        visual_description="a flat plate",
        overall_envelope_mm=envelope,
        features=list(features)
        or [BriefFeature(name="body", purpose="bulk", approx_dims_mm="40x40x10", method="box")],
        min_wall_mm=2.0,
        print_orientation_note="flat side down",
    )


# --- stated_numbers: what earns protection --------------------------------------------------------


def test_prompt_dimensional_numbers_qualify_bare_integers_do_not() -> None:
    prompt = (
        "A 36-tooth timing gear, about 76 mm tip diameter, 12 mm bore, boss OD 22 mm, "
        "six 8 mm lightening holes; mates a pinion with 19 teeth"
    )
    s = stated_numbers(prompt, _brief())
    assert {76.0, 12.0, 22.0, 8.0} <= s
    assert 36.0 not in s and 19.0 not in s  # tooth counts are structure, never pinned
    assert 6.0 not in s  # "six" spelled out; bare counts stay free


def test_wall_floor_is_excluded_and_dia_context_matches() -> None:
    # the wall floor is a FLOOR, not a dimension -- pinning it would fight the wall repair.
    s = stated_numbers("2 mm walls, bore of 12", _brief())
    assert 2.0 not in s
    assert 12.0 in s  # "bore of 12" -- dia context without a mm suffix still qualifies


def test_dimension_chains_pin_every_number() -> None:
    # the measured M4 gap: a unit-bearing chain pinned only the number carrying the unit --
    # hook_j's 50 (in "back plate 30 x 50 x 4 mm") drifted to 62.70 under ACCEPTED.
    s = stated_numbers("back plate 30 x 50 x 4 mm, hook horizontal reach 40 mm", _brief())
    assert {30.0, 50.0, 4.0, 40.0} <= s
    # hook_coat's chain: 80 and 40 now qualify (only the 4 did before)
    s2 = stated_numbers("back plate 80 x 40 x 4 mm", _brief())
    assert {80.0, 40.0, 4.0} <= s2
    # a chain with NO trailing unit is not dimensional -- bare counts stay free
    assert 3.0 not in stated_numbers("a 3 x 3 grid of sockets", _brief())


def test_angles_qualify() -> None:
    # hook_coat's measured drift: "30 degrees upward angle" -> built 33.4 under ACCEPTED.
    s = stated_numbers(
        "two hooks extending forward 50 mm each at 30 degrees upward angle", _brief()
    )
    assert {50.0, 30.0} <= s
    assert 45.0 in stated_numbers("a 45° chamfer on the rim", _brief())
    assert 180.0 in stated_numbers("two M3 set-screw radial holes at 180 degrees", _brief())
    assert 90.0 not in stated_numbers("rotate it by ninety", _brief())  # words stay free


def test_registry_and_intake_sources_qualify() -> None:
    brief = _brief(
        BriefFeature(
            name="bolt holes", purpose="mounting", approx_dims_mm="M5", method="bolt_holes", count=4
        )
    )
    s = stated_numbers("a bracket", brief)
    assert 5.5 in s  # M5 clearance, ISO 273 -- registry-resolved, lands in PARAMS as 5.5
    s2 = stated_numbers(
        "a bracket", brief, extra_context="- dimension: mount width = 120 mm (source: user)\n"
    )
    assert 120.0 in s2
    # the model's own brief numbers never enter (anchor invariant)
    brief2 = _brief(
        BriefFeature(name="rib", purpose="stiffness", approx_dims_mm="4.7 thick", method="box")
    )
    assert 4.7 not in stated_numbers("a bracket", brief2)


# --- the pure non-degrade judgment ----------------------------------------------------------------

BORE = _HoleSpec(label="bore", count=1, target_d_mm=12.0, source="stated in the brief")
M5 = _HoleSpec(label="bolt holes", count=4, target_d_mm=5.5, source="M5 clearance, ISO 273")


def test_judge_rejects_the_incident_bore_drift() -> None:
    # the measured incident: baseline exact, winner 12->12.5 (dev 0.5 > max(2%*12, 0.2) = 0.24).
    ok, notes = _judge_bores([BORE], {"bore": 0.0}, {"bore": 0.5})
    assert ok is False
    assert any("rejected" in n for n in notes)


def test_judge_non_degrade_allows_preexisting_author_error() -> None:
    # baseline already deviated 0.4 (author error, the revision loop's job) -- an equal winner
    # is NOT the polish's fault and passes; getting WORSE than the baseline still rejects.
    ok, _ = _judge_bores([BORE], {"bore": 0.4}, {"bore": 0.4})
    assert ok is True
    ok2, _ = _judge_bores([BORE], {"bore": 0.4}, {"bore": 0.55})
    assert ok2 is False


def test_judge_differential_measurability_rejects_erasure() -> None:
    # measurable on the baseline, gone on the winner = the fit erased it. Never the cheap move.
    ok, notes = _judge_bores([BORE], {"bore": 0.0}, {"bore": None})
    assert ok is False
    assert any("erased" in n for n in notes)
    # unmeasurable on BOTH: no claim, no rejection (and the note says so).
    ok2, notes2 = _judge_bores([BORE], {"bore": None}, {"bore": None})
    assert ok2 is True
    assert any("no claim" in n for n in notes2)


def test_judge_registry_targets_get_the_tight_band() -> None:
    ok, _ = _judge_bores([M5], {"bolt holes": 0.0}, {"bolt holes": 0.15})
    assert ok is False  # 0.15 > the 0.1 registry band
    ok2, _ = _judge_bores([M5], {"bolt holes": 0.0}, {"bolt holes": 0.08})
    assert ok2 is True


def test_judge_envelope_rejects_the_incident_tip_growth() -> None:
    # the incident's derived channel: module drift grew the tip 76->92.4 -- the envelope check
    # is the MEASURED cover for it (the pins cannot see a derived dimension).
    env = (76.0, 76.0, 14.0)
    ok, _ = _judge_envelope(env, (14.0, 76.0, 76.0), (14.0, 76.0, 92.4))
    assert ok is False
    ok2, _ = _judge_envelope(env, (14.0, 76.0, 76.0), (14.0, 76.5, 76.5))
    assert ok2 is True  # inside the 2% band
    # non-degrade: a baseline already 4mm off allows an equally-off winner.
    ok3, _ = _judge_envelope(env, (14.0, 76.0, 80.0), (14.0, 76.0, 80.0))
    assert ok3 is True


# --- the audit shell: vacuous pass, lenient baseline, fail-closed error ---------------------------


def test_audit_vacuous_pass_when_nothing_is_stated(tmp_path: Path) -> None:
    brief = _brief(envelope="roughly palm sized")  # unparseable envelope, no sized holes
    stl = tmp_path / "w.stl"
    trimesh.creation.box(extents=(40, 40, 10)).export(str(stl))
    audit = stated_dim_audit(brief, None, tmp_path / "w.step", None, stl)
    assert audit.ok is True and audit.error is False
    assert audit.notes == ["stated-dim audit: no stated targets"]


def test_audit_envelope_runs_with_unreadable_baseline(tmp_path: Path) -> None:
    # a lenient baseline never blocks: winner judged against the 2% band alone.
    brief = _brief(envelope="76 x 76 x 14")
    win = tmp_path / "win.stl"
    trimesh.creation.box(extents=(76, 76, 14)).export(str(win))
    audit = stated_dim_audit(brief, None, tmp_path / "w.step", None, win)
    assert audit.ok is True
    assert any("envelope: held" in n for n in audit.notes)


def test_audit_fails_closed_when_the_winner_is_unverifiable(tmp_path: Path) -> None:
    brief = _brief(envelope="76 x 76 x 14")
    audit = stated_dim_audit(brief, None, tmp_path / "w.step", None, tmp_path / "missing.stl")
    assert audit.ok is False and audit.error is True
    assert any("failed to run" in n for n in audit.notes)
