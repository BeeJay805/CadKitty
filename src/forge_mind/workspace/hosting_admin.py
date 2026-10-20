"""Offline host maintenance. These commands never start a provider or listener."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge_mind.workspace.hosting import HostConfig, HostLock
from forge_mind.workspace.hosting_backup import restore, snapshot, verify
from forge_mind.workspace.hosting_budget import grant
from forge_mind.workspace.hosting_identity import canonical_principal
from forge_mind.workspace.hosting_store import HostingStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("--root", type=Path, required=True)
    capture.add_argument("--destination", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--snapshot", type=Path, required=True)
    recover = commands.add_parser("restore")
    recover.add_argument("--snapshot", type=Path, required=True)
    recover.add_argument("--destination", type=Path, required=True)
    allowance = commands.add_parser("grant")
    allowance.add_argument(
        "--subject", required=True, help="Validated subject from the configured issuer; never email"
    )
    allowance.add_argument("--grant-id", required=True)
    allowance.add_argument("--seconds", type=int, required=True)
    fence = commands.add_parser("recover-fenced")
    fence.add_argument("--old-host", required=True)
    fence.add_argument("--job-id", action="append", default=[])
    fence.add_argument("--evidence", type=Path, required=True)
    fence.add_argument("--restore-id")
    clear = commands.add_parser("clear-runner-hold")
    clear.add_argument("--receipt", type=Path, required=True)
    dispose = commands.add_parser("dispose-restored-runner-hold")
    dispose.add_argument("--restore-id", required=True)
    dispose.add_argument("--hold-id", action="append", required=True)
    dispose.add_argument("--operation-id", required=True)
    dispose.add_argument("--old-fence", required=True)
    dispose.add_argument("--old-image", required=True)
    dispose.add_argument("--job-id", action="append", default=[])
    dispose.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args.root, args.destination)
    elif args.command == "verify":
        result = verify(args.snapshot)
    elif args.command == "restore":
        result = restore(args.snapshot, args.destination)
    else:
        config = HostConfig.from_env()
        # Construct/migrate only while excluded from the serving process.
        with HostLock(config.root):
            store = HostingStore(config)
            if args.command == "grant":
                owner = canonical_principal(config.issuer, args.subject)
                with store.connect() as con:
                    con.execute("BEGIN IMMEDIATE")
                    grant(con, owner, args.grant_id, args.seconds)
                result = {
                    "granted_seconds": args.seconds,
                    "scope": "operator_compute_allowance_not_money",
                }
        if args.command == "recover-fenced":
            result = store.apply_fenced_recovery(
                old_host=args.old_host,
                job_ids=args.job_id,
                fencing_evidence=args.evidence,
                restore_id=args.restore_id,
            )
        elif args.command == "clear-runner-hold":
            result = store.clear_runner_hold(args.receipt)
        elif args.command == "dispose-restored-runner-hold":
            result = store.dispose_restored_runner_hold(
                restore_id=args.restore_id,
                hold_ids=args.hold_id,
                operation_id=args.operation_id,
                old_fence=args.old_fence,
                old_image=args.old_image,
                job_ids=args.job_id,
                fencing_evidence=args.evidence,
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
