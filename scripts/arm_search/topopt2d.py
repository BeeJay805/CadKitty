"""Thin CLI over ``forge_mind.structures.topopt`` (the code moved into the engine; ADR-0097)."""

from __future__ import annotations

import sys

# Re-exports for callers that still import this script as a module.
from forge_mind.structures.topopt import (  # noqa: F401
    Domain,
    as_ascii,
    cantilever,
    optimize,
)


def main() -> int:
    res = optimize(cantilever(), max_iter=50)
    from forge_mind.structures.topopt import as_ascii as _ascii

    print(_ascii(res["density"]))
    print(
        f"\niters={res['iters']}  compliance {res['compliance'][0]:.1f} -> {res['compliance'][-1]:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
