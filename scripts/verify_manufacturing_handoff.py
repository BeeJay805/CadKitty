"""Verify a downloaded manufacturing review ZIP without extracting its contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge_mind.workspace.handoff import verify_manufacturing_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    try:
        result = verify_manufacturing_handoff(args.package)
    except ValueError as exc:
        print(json.dumps({"integrity": "invalid", "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
