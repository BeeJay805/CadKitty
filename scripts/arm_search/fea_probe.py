"""Thin CLI over ``forge_mind.structures.loadcase`` (the code moved into the engine; ADR-0097).

Usage:
  uv run --no-sync python scripts/arm_search/fea_probe.py --smoke
  uv run --no-sync python scripts/arm_search/fea_probe.py --stl path/to/link.stl
"""

from __future__ import annotations

import argparse
import json
import sys

# Re-exports for callers that still import this script as a module.
from forge_mind.structures.loadcase import (  # noqa: F401
    G_M_S2,
    SMOKE_CANTILEVER_5N,
    LoadCase,
    canonical_link_case,
    canonical_link_orient,
    probe_env,
    run_load_case,
)


def _smoke() -> int:
    import trimesh

    env = probe_env()
    if not env["fea_is_validated"]:
        print(json.dumps({"ok": False, "reason": "fea_not_validated", **env}, indent=2))
        return 1
    bar = trimesh.creation.box(extents=(10.0, 10.0, 60.0))
    lr = run_load_case(bar, SMOKE_CANTILEVER_5N)
    if lr is None:
        print(json.dumps({"ok": False, "reason": "load_response_none", **env}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "fea_is_validated": True,
                "smoke_case": SMOKE_CANTILEVER_5N.name,
                "max_deflection_mm": lr.max_deflection_mm,
                "max_von_mises_mpa": lr.max_von_mises_mpa,
                "load_case": lr.load_case,
            },
            indent=2,
        )
    )
    return 0


def _stl(path: str) -> int:
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    lr = run_load_case(mesh, SMOKE_CANTILEVER_5N)
    print(
        json.dumps(
            {
                "ok": lr is not None,
                "stl": path,
                "max_deflection_mm": None if lr is None else lr.max_deflection_mm,
                "max_von_mises_mpa": None if lr is None else lr.max_von_mises_mpa,
            },
            indent=2,
        )
    )
    return 0 if lr is not None else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validated-FEA load-case probe.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--stl", metavar="PATH")
    args = ap.parse_args(argv)
    return _smoke() if args.smoke else _stl(args.stl)


if __name__ == "__main__":
    sys.exit(main())
