"""structures.loadcase: LoadCase wiring, canonical orientation, fail-closed passthrough.

Ported from tests/unit/arm_search/test_fea_probe.py when the code was promoted into the engine
(ADR-0097 consolidation); same assertions, real imports. The real-FEA smoke is marked slow.
"""

from __future__ import annotations

import pytest

from forge_mind.structures import loadcase as lc


def test_load_case_defaults() -> None:
    c = lc.LoadCase(name="x", force_n=5.0)
    assert c.load_dir == (0.0, 0.0, -1.0)
    assert c.orient is None


def test_run_load_case_passes_through_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-closed passthrough: no number from the solver -> no number from the case."""
    trimesh = pytest.importorskip("trimesh")
    monkeypatch.setattr(lc, "estimate_load_response", lambda *a, **k: None)
    bar = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    assert lc.run_load_case(bar, lc.SMOKE_CANTILEVER_5N) is None


def test_run_load_case_forwards_load_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5 threading: LoadCase.load_patch reaches estimate_load_response; None (the default) is
    forwarded verbatim so the whole-face behaviour stays byte-identical."""
    trimesh = pytest.importorskip("trimesh")
    seen: dict[str, object] = {}

    def _fake(mesh: object, force_n: float, **kw: object) -> None:
        seen["load_patch"] = kw.get("load_patch")
        return None

    monkeypatch.setattr(lc, "estimate_load_response", _fake)
    bar = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    lc.run_load_case(bar, lc.SMOKE_CANTILEVER_5N)
    assert seen["load_patch"] is None
    patched = lc.LoadCase(name="p", force_n=5.0, load_patch=((3.0, 1.5), 2.0))
    lc.run_load_case(bar, patched)
    assert seen["load_patch"] == ((3.0, 1.5), 2.0)


def test_describe_load_case_shows_patch() -> None:
    """The human-units echo names a patch (Fable's load-case honesty mitigation extends to WHERE
    the load lands, not just how big it is)."""
    face = lc.LoadCase(name="c", force_n=4.9)
    assert "over the whole loaded face" in lc.describe_load_case(face)
    patch = lc.LoadCase(name="c", force_n=4.9, load_patch=((10.0, 2.0), 1.5))
    echo = lc.describe_load_case(patch)
    assert "patch (r=1.5 mm)" in echo and "(10.0, 2.0)" in echo


def test_run_load_case_applies_orient(monkeypatch: pytest.MonkeyPatch) -> None:
    """orient reorients the mesh (fixed port -> z-min) before the solve, on a copy."""
    trimesh = pytest.importorskip("trimesh")
    import numpy as np
    from trimesh.transformations import rotation_matrix

    captured: dict[str, float] = {}

    def _fake(mesh: object, force_n: float, **kw: object) -> None:
        bounds = mesh.bounds  # type: ignore[attr-defined]
        captured["z_extent"] = float(bounds[1][2] - bounds[0][2])
        captured["force_n"] = float(force_n)
        return None

    monkeypatch.setattr(lc, "estimate_load_response", _fake)
    bar = trimesh.creation.box(extents=(10.0, 10.0, 60.0))
    rot = rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    case = lc.LoadCase(name="rot", force_n=3.0, orient=tuple(map(tuple, rot)))
    assert lc.run_load_case(bar, case) is None
    assert abs(captured["z_extent"] - 10.0) < 1e-6
    assert captured["force_n"] == 3.0
    assert abs((bar.bounds[1][2] - bar.bounds[0][2]) - 60.0) < 1e-6  # caller's mesh untouched


def test_canonical_orient_stands_link_up() -> None:
    trimesh = pytest.importorskip("trimesh")
    import numpy as np

    bar = trimesh.creation.box(extents=(100.0, 20.0, 8.0))
    m = bar.copy()
    m.apply_transform(np.asarray(lc.canonical_link_orient(bar), dtype=float))
    assert int(np.argmax(m.extents)) == 2  # longest axis is now Z
    assert abs(m.bounds[0][2]) < 1e-6  # base sits at z=0


def test_canonical_case_loads_the_weak_axis() -> None:
    trimesh = pytest.importorskip("trimesh")

    case_a = lc.canonical_link_case(trimesh.creation.box(extents=(100.0, 20.0, 8.0)))
    assert case_a.load_dir == (1.0, 0.0, 0.0)
    assert abs(case_a.force_n - 0.5 * lc.G_M_S2) < 1e-9
    case_b = lc.canonical_link_case(trimesh.creation.box(extents=(100.0, 8.0, 20.0)))
    assert case_b.load_dir == (0.0, 1.0, 0.0)


def test_canonical_case_is_deterministic() -> None:
    trimesh = pytest.importorskip("trimesh")
    bar = trimesh.creation.box(extents=(90.0, 22.0, 10.0))
    c1 = lc.canonical_link_case(bar, payload_kg=0.7)
    c2 = lc.canonical_link_case(bar, payload_kg=0.7)
    assert c1 == c2
    assert abs(c1.force_n - 0.7 * lc.G_M_S2) < 1e-9


@pytest.mark.slow
def test_smoke_real_fea_positive_deflection() -> None:
    pytest.importorskip("skfem")
    pytest.importorskip("pytetwild")
    trimesh = pytest.importorskip("trimesh")
    env = lc.probe_env()
    assert env["fea_is_validated"] is True
    bar = trimesh.creation.box(extents=(10.0, 10.0, 60.0))
    lr = lc.run_load_case(bar, lc.SMOKE_CANTILEVER_5N)
    assert lr is not None
    assert lr.max_deflection_mm > 0.0
    assert lr.max_von_mises_mpa > 0.0
