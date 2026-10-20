"""Staged construction (complex-parts arc P1): the seam's laws, pinned.

- FORGE_STAGED_BUILD default OFF (flag-off parity is the whole suite passing unchanged).
- Checkpoint checks catch erasure (census/volume non-regression) and verify the stage's
  own feature by measurement where measurable.
- A refused stage hands back the last GOOD partial (never a hard fail).
The end-to-end tests drive staged_initial with SCRIPTED authors through the REAL sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.agentic import staged as staged_mod
from forge_mind.agentic.staged import (
    _census_regressions,
    _feature_expectation,
    _prefix_kept,
    staged_enabled,
    staged_initial,
    staged_min_features,
)

BASE_CODE = """from build123d import *
PARAMS = {"plate_len": 60.0, "plate_wid": 40.0, "plate_thk": 8.0}
with BuildPart() as part:
    Box(PARAMS["plate_len"], PARAMS["plate_wid"], PARAMS["plate_thk"])
result = part.part
"""

HOLES_CODE = """from build123d import *
PARAMS = {"plate_len": 60.0, "plate_wid": 40.0, "plate_thk": 8.0, "bolt_d": 4.4, "bolt_cc": 40.0}
with BuildPart() as part:
    Box(PARAMS["plate_len"], PARAMS["plate_wid"], PARAMS["plate_thk"])
    with Locations((-PARAMS["bolt_cc"] / 2, 0, 0), (PARAMS["bolt_cc"] / 2, 0, 0)):
        Hole(radius=PARAMS["bolt_d"] / 2, depth=PARAMS["plate_thk"] + 2)
result = part.part
"""

ERASED_CODE = """from build123d import *
PARAMS = {"tiny": 4.0}
with BuildPart() as part:
    Box(PARAMS["tiny"], PARAMS["tiny"], PARAMS["tiny"])
result = part.part
"""


def _brief(features: list[SimpleNamespace]) -> SimpleNamespace:
    b = SimpleNamespace(
        object_name="test plate",
        overall_envelope_mm="60 x 40 x 8 mm",
        features=features,
    )
    b.model_copy = lambda update=None: SimpleNamespace(
        object_name=b.object_name,
        overall_envelope_mm=b.overall_envelope_mm,
        features=(update or {}).get("features", []),
    )
    return b


def _scripted_authors(monkeypatch, codes: list[str]) -> None:
    """author_initial serves codes[0]; author_revision serves codes[1:] in order (repeating
    the last on retries)."""
    calls = {"n": 0}

    def _initial(brief, **kw):
        return SimpleNamespace(code=codes[0], approach="scripted"), SimpleNamespace(cost_usd=0.0)

    def _revision(brief, prior, feedback, **kw):
        calls["n"] += 1
        ix = min(calls["n"], len(codes) - 1)
        return SimpleNamespace(code=codes[ix], approach="scripted"), SimpleNamespace(cost_usd=0.0)

    monkeypatch.setattr(staged_mod, "author_initial", _initial)
    monkeypatch.setattr(staged_mod, "author_revision", _revision)


def test_flag_default_off_and_min_features(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_STAGED_BUILD", raising=False)
    assert staged_enabled() is False
    monkeypatch.setenv("FORGE_STAGED_BUILD", "1")
    assert staged_enabled() is True
    monkeypatch.setenv("FORGE_STAGED_MIN_FEATURES", "7")
    assert staged_min_features() == 7


def test_census_regressions_and_prefix() -> None:
    assert _census_regressions({"4.5": 2}, {"4.5": 2}) == []
    assert _census_regressions({"4.5": 2}, {"4.5": 1}) == ["holes at ~4.5mm dropped 2 -> 1"]
    assert _prefix_kept(BASE_CODE, HOLES_CODE.replace("    with Locations", "    with Locations"))
    # params-dict growth does not break the lock; body rewrites do
    assert _prefix_kept(BASE_CODE, HOLES_CODE)
    assert not _prefix_kept(HOLES_CODE, BASE_CODE)


def test_feature_expectation_parses_hole_features() -> None:
    d, n = _feature_expectation(
        SimpleNamespace(
            name="mounting holes",
            purpose="fastening",
            approx_dims_mm="4.4 mm diameter through",
            count=2,
        )
    )
    assert d == 4.4 and n == 2
    d, n = _feature_expectation(
        SimpleNamespace(name="stiffening rib", purpose="rigidity", approx_dims_mm="3 mm thick")
    )
    assert d is None and n == 1


def test_staged_builds_base_then_verified_hole_stage(monkeypatch, tmp_path: Path) -> None:
    features = [
        SimpleNamespace(
            name="bolt holes",
            purpose="fastening",
            approx_dims_mm="4.4 mm diameter, 2 holes",
            count=2,
        )
    ]
    _scripted_authors(monkeypatch, [BASE_CODE, HOLES_CODE])
    res = staged_initial(_brief(features), work_dir=tmp_path, part_name="tp")
    assert res.code is not None and "Hole" in res.code
    assert res.completed == 2 and res.total_stages == 2
    assert res.records[0].name == "__base__" and res.records[0].built
    assert res.records[1].built and res.records[1].feature_verified == "measured"
    assert res.records[1].census.get("4.5") == 2  # 4.4 bins to 4.5 at 0.5mm
    stages = json.loads((tmp_path / "stages.json").read_text(encoding="utf-8"))
    assert len(stages) == 2


def test_staged_refuses_an_erasing_stage_and_hands_back_the_partial(
    monkeypatch, tmp_path: Path
) -> None:
    features = [
        SimpleNamespace(name="lightening", purpose="mass", approx_dims_mm="pocket", count=1)
    ]
    _scripted_authors(monkeypatch, [BASE_CODE, ERASED_CODE])
    res = staged_initial(_brief(features), work_dir=tmp_path, part_name="tp")
    # the erasing stage fails its volume-floor checkpoint on every try -> refused,
    # and the BASE code is handed back (the standard loop finishes the part)
    assert res.records[1].refused and "erased" in (res.records[1].issue or "")
    assert res.completed == 1
    assert res.code is not None and "Hole" not in res.code


def test_hole_census_regression_blocks_a_stage(monkeypatch, tmp_path: Path) -> None:
    # stage 1 adds holes; stage 2's candidate REMOVES them (BASE_CODE) -> census regression
    features = [
        SimpleNamespace(
            name="bolt holes",
            purpose="fastening",
            approx_dims_mm="4.4 mm diameter, 2 holes",
            count=2,
        ),
        SimpleNamespace(name="chamfer", purpose="finish", approx_dims_mm="1 mm edge break"),
    ]
    _scripted_authors(monkeypatch, [BASE_CODE, HOLES_CODE, BASE_CODE])
    res = staged_initial(_brief(features), work_dir=tmp_path, part_name="tp")
    assert res.records[1].built
    assert res.records[2].refused and "regressed" in (res.records[2].issue or "")
    assert res.code is not None and "Hole" in res.code  # the good stage survives


@pytest.mark.parametrize("flag", ["", "0", "off"])
def test_disabled_values(monkeypatch, flag: str) -> None:
    monkeypatch.setenv("FORGE_STAGED_BUILD", flag)
    assert staged_enabled() is False


def test_positive_cylindrical_bodies_are_not_hole_expectations() -> None:
    # the dense_gearbox_face first-contact bug: a DISC BODY with "110 mm diameter" must
    # not demand a 110mm hole; only hole-like names/purposes set census expectations
    d, _ = _feature_expectation(
        SimpleNamespace(
            name="main disc body",
            purpose="structure",
            approx_dims_mm="110 mm diameter x 8 mm thick",
        )
    )
    assert d is None
    d, n = _feature_expectation(
        SimpleNamespace(
            name="dowel holes", purpose="alignment", approx_dims_mm="8 mm diameter", count=3
        )
    )
    assert d == 8.0 and n == 3


def test_body_features_fold_into_the_base_stage(monkeypatch, tmp_path: Path) -> None:
    # Row-53 lesson: a 'box shell' must NOT get its own stage (no-op / coplanar re-fuse);
    # it is authored WITH the base. Only the holes get a stage here.
    from forge_mind.agentic.staged import _is_body_feature

    shell = SimpleNamespace(name="box shell", purpose="enclosure", approx_dims_mm="2.5 mm walls")
    holes = SimpleNamespace(
        name="bolt holes", purpose="fastening", approx_dims_mm="4.4 mm diameter, 2 holes", count=2
    )
    assert _is_body_feature(shell) and not _is_body_feature(holes)
    _scripted_authors(monkeypatch, [BASE_CODE, HOLES_CODE])
    res = staged_initial(_brief([shell, holes]), work_dir=tmp_path, part_name="tp")
    assert res.total_stages == 2  # base(+shell) and the holes -- NOT 3
    assert res.records[0].name.startswith("__base__") and "box shell" in res.records[0].name
    assert res.records[1].name == "bolt holes" and res.records[1].built
    assert res.completed == 2
