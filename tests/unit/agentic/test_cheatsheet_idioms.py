"""EVERY cheatsheet idiom executes through the REAL sandbox (the lever-b verification
rule, and the promise the cheatsheet docstring makes): the author reads only API that is
PROVEN to run on the installed kernel. A new idiom that does not build fails here, so an
unverified snippet can never reach the author's prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic.cheatsheet import VERIFIED_IDIOMS, author_skills_block
from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code

pytestmark = pytest.mark.slow  # each case cold-starts the kernel subprocess


@pytest.mark.parametrize("name", sorted(VERIFIED_IDIOMS))
def test_idiom_executes_in_the_sandbox(name: str, tmp_path: Path) -> None:
    res = run_build123d_code(
        VERIFIED_IDIOMS[name], tmp_path, part_name=name, limits=SandboxLimits(timeout_s=180)
    )
    assert res.ok, f"idiom {name!r} failed: {res.error_type}: {res.error_message}"
    assert res.stl_path is not None and res.stl_path.stat().st_size > 0


def test_skills_block_includes_the_new_pattern_idioms() -> None:
    block = author_skills_block()
    for key in (
        "polar_holes",
        "linear_rib_array",
        "mirror_feature",
        "counterbore_holes",
        "helix_coil",
        "radial_set_screw_holes",
        "external_iso_thread",
        "internal_iso_thread",
        "involute_spur_gear",
        "tapered_limb_shell",
        "joint_module",
        "servo_pocket",
        "backward_curved_impeller",
        "crank_web_counterweight",
        "tapered_fin_stack",
        "chamfered_bore_entry",
    ):
        assert key.replace("_", " ") in block or key in block
    assert "PolarLocations" in block and "Helix(" in block
    assert "Ellipse(" in block and "Sphere(" in block  # the limb-shell + joint-module vocabulary
    assert "49.5, 10.0" in block  # the servo lug pattern rides the executed snippet
