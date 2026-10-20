"""Package nominal workspace evidence and all eight idler corner revisions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.engineering_service import release_package, report_path
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.store import Store


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("Review package evidence is incomplete or inconsistent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new package filename")
    root = Path(__file__).resolve().parents[1]
    session = json.loads((args.run / "session.json").read_text(encoding="utf-8"))
    nominal = json.loads((args.run / "rebuild.json").read_text(encoding="utf-8"))
    mechanical = json.loads((args.run / "mechanical.json").read_text(encoding="utf-8"))
    corners = json.loads((args.run / "corners.json").read_text(encoding="utf-8"))
    owner, sid, rid = "local-owner", session["id"], nominal["id"]
    service = WorkspaceService(Store(args.store), Executor("trusted_local"))
    revision = service._load_revision(sid, rid)
    require(revision["measurement"]["engine_fingerprint"] == engine_fingerprint())
    require(mechanical["step_sha256"] == revision["measurement"]["artifact_hashes"]["part.step"])
    require(
        mechanical["checker_sha256"]
        == hashlib.sha256((root / "scripts/idler_mechanical_check.py").read_bytes()).hexdigest()
    )
    for key, count in [("checks", 45), ("tool_access", 9), ("rotation_bounds", 2)]:
        require(
            len(mechanical[key]) == count and all(c["status"] == "pass" for c in mechanical[key])
        )
    require(
        len(mechanical["disassembly"]) == 3
        and all(s["sampled_clear"] for s in mechanical["disassembly"])
    )
    expected = {
        (e, x, -e if closed else 0.0)
        for e, x, closed in itertools.product([0.15, 0.35], [-15.0, 15.0], [False, True])
    }
    actual = {
        (
            s["variables"]["screw_endplay"],
            s["variables"]["carriage_x"],
            s["variables"]["screw_float"],
        )
        for s in corners
    }
    require(len(corners) == 8 and actual == expected)
    for sample in corners:
        require(sample["geometry_valid"] and sample["interference"]["status"] == "pass")
        require(all(f["status"] != "fail" for f in sample["findings"]))
        require(all(m["status"] == "pass" for m in sample["interfaces"]))
    core = release_package(service, sid, owner, rid)
    files = {"workspace-review.zip": core}
    for name in [
        "mechanical.json",
        "corners.json",
        "study.json",
        "rebuild.json",
        "source-contract.json",
        "contract.json",
    ]:
        files[name] = args.run / name
    for name in [
        "docs/idler-mechanical-review.md",
        "docs/decision_log/0138-idler-mechanical-closure.md",
        "docs/benchmarks/engineering-idler/source-revision.json",
        "docs/benchmarks/engineering-idler/mechanical-plan.json",
        "scripts/idler_mechanical_revision.py",
        "scripts/idler_mechanical_check.py",
        "scripts/idler_mechanical_bench.py",
    ]:
        files["reproduction/" + name] = root / name
    live = service.store.get(sid, owner)
    for sample in corners:
        candidate = service._load_revision(sid, sample["revision_id"])
        require(candidate["measurement"]["engine_fingerprint"] == engine_fingerprint())
        for name, digest in candidate["measurement"]["artifact_hashes"].items():
            path = service.store.directory(sid, candidate["id"]) / name
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest)
            files[f"corners/{candidate['id']}/{name}"] = path
        receipt = next(
            r
            for r in live["reports"]
            if r["revision_id"] == candidate["id"]
            and r["engine_fingerprint"] == engine_fingerprint()
            and r["input_version"] == live["engineering"]["version"]
        )
        for name in receipt["hashes"]:
            files[f"corners/{candidate['id']}/engineering/{name}"] = report_path(
                service, sid, owner, receipt["id"], name
            )
    manifest = {
        "session": sid,
        "nominal_revision": rid,
        "engine_fingerprint": engine_fingerprint(),
        "disposition": "Review evidence only. Manufacturing, strength and physical operation remain unverified.",
        "files": {},
    }
    with ZipFile(args.output, "w", ZIP_DEFLATED) as archive:
        for name, path in files.items():
            data = path.read_bytes()
            archive.writestr(name, data)
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    for path in [core, args.output]:
        with ZipFile(path) as archive:
            recorded = json.loads(archive.read("manifest.json"))["files"]
            require(set(recorded) == set(archive.namelist()) - {"manifest.json"})
            require(
                all(
                    (
                        hashlib.sha256(archive.read(name)).hexdigest() == digest
                        for name, digest in recorded.items()
                    )
                )
            )
            require(archive.testzip() is None)
    result = {
        "output": str(args.output),
        "members_verified": len(files),
        "core_package": str(core),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "session": sid,
        "revision": rid,
    }
    (args.run / "package-verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
