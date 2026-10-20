"""Small Linux isolation probe used only by explicit worker smoke qualification."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HOST_PATHS = ("/root/.codex", "/mnt/c/Users", "/var/run/docker.sock")


def path_observation(path: Path) -> dict[str, str]:
    try:
        path.stat()
    except FileNotFoundError:
        return {"state": "absent"}
    except PermissionError as exc:
        return {"state": "denied", "error": type(exc).__name__}
    except OSError as exc:
        return {"state": "error", "error": type(exc).__name__}
    return {"state": "present"}


def main(request: Path) -> None:
    payload = json.loads(request.read_text(encoding="utf-8"))
    if payload.get("sleep"):
        time.sleep(float(payload["sleep"]))
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    routes = Path("/proc/net/route").read_text().splitlines()[1:]
    observations = {value: path_observation(Path(value)) for value in HOST_PATHS}
    result = {
        "unprivileged": os.getuid() == 65534 and os.getgid() == 65534,  # type: ignore[attr-defined]
        "readonly_root": bool(os.statvfs("/").f_flag & os.ST_RDONLY),  # type: ignore[attr-defined]
        "no_capabilities": int(status["CapEff"].strip(), 16) == 0,
        "no_new_privileges": status["NoNewPrivs"].strip() == "1",
        "no_network_route": not routes,
        "host_paths_absent": all(value["state"] == "absent" for value in observations.values()),
        "host_paths_inaccessible": all(
            value["state"] in {"absent", "denied"} for value in observations.values()
        ),
        "host_path_observations": observations,
    }
    (request.parent / "probe-result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
