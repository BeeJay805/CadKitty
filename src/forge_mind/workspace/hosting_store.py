"""Host-only Store integration; every entrypoint retains one transaction authority."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import socket
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from forge_mind.workspace.hosting import HostConfig, HostingError, HostLock, write_json
from forge_mind.workspace.hosting_backup import _fenced_plan_locked, current_restore
from forge_mind.workspace.hosting_budget import initialize, reserve, settle
from forge_mind.workspace.hosting_runner import RunnerUncertainError
from forge_mind.workspace.store import Store, WorkspaceError


class HostingStore(Store):
    def __init__(self, config: HostConfig) -> None:
        self.host_config = config
        self._hosting_ready = False
        self._active = threading.local()
        super().__init__(config.root)
        with super().connect() as con:
            initialize(con)
            con.execute(
                "CREATE TABLE IF NOT EXISTS host_holds (job_id TEXT PRIMARY KEY, reason TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS host_runner_holds (hold_id TEXT PRIMARY KEY, reason TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS host_runner_hold_evidence (hold_id TEXT PRIMARY KEY, document TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS host_hold_dispositions (id TEXT PRIMARY KEY, document TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS host_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), issuer TEXT NOT NULL)"
            )
            identity = con.execute("SELECT issuer FROM host_identity WHERE singleton=1").fetchone()
            if identity is None:
                if con.execute("SELECT 1 FROM sessions LIMIT 1").fetchone():
                    raise HostingError(
                        "Private host requires a fresh Store or an explicit reviewed owner migration"
                    )
                con.execute("INSERT INTO host_identity VALUES (1,?)", (config.issuer,))
            elif identity[0] != config.issuer:
                raise HostingError(
                    "Store owner issuer differs from the configured canonical issuer"
                )
            if any(
                re.fullmatch(r"principal:v1:[a-f0-9]{64}", row[0]) is None
                for row in con.execute("SELECT DISTINCT owner FROM sessions")
            ):
                raise HostingError("Private host cannot silently remap legacy raw-subject owners")
        self._hosting_ready = True

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with super().connect() as con:
            yield con
            if self._hosting_ready and con.in_transaction:
                self._settle_terminal(con)

    def _settle_terminal(self, con: sqlite3.Connection) -> None:
        rows = con.execute(
            "SELECT r.job_id,r.owner,r.seconds,j.started,j.state FROM host_reservations r JOIN jobs j ON j.id=r.job_id WHERE r.used_seconds IS NULL AND j.state NOT IN ('queued','running') AND NOT EXISTS (SELECT 1 FROM host_holds h WHERE h.job_id=r.job_id)"
        ).fetchall()
        for row in rows:
            # A confirmed orphan's exact stop time is unknown. Preserve the
            # conservative original reservation rather than inventing a refund.
            used = (
                row[2]
                if row[4] == "interrupted"
                else 0
                if row[3] is None
                else min(row[2], math.ceil(max(0, time.time() - row[3])))
            )
            settle(con, row[1], row[0], used)

    def _claim_allowed(self, con: sqlite3.Connection) -> bool:
        return (
            con.execute("SELECT 1 FROM host_holds LIMIT 1").fetchone() is None
            and con.execute("SELECT 1 FROM host_runner_holds LIMIT 1").fetchone() is None
        )

    def _enqueue_with_connection(
        self,
        con: sqlite3.Connection,
        sid: str,
        owner: str,
        payload: dict[str, Any],
        *,
        construction_approval: bool = False,
    ) -> str:
        if re.fullmatch(r"principal:v1:[a-f0-9]{64}", owner) is None:
            raise WorkspaceError("Private host requires a validated canonical owner")
        if con.execute("SELECT 1 FROM host_runner_holds LIMIT 1").fetchone():
            raise WorkspaceError(
                "Runner uncertainty requires offline reconciliation before admission"
            )
        if "host_lineage_id" in payload:
            raise WorkspaceError("Host compute lineage is server-owned")
        prior = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        previous_id = json.loads(prior[0]).get("job_id") if prior else None
        previous = con.execute(
            "SELECT payload FROM jobs WHERE id=? AND session=?", (previous_id, sid)
        ).fetchone()
        previous_payload = json.loads(previous[0]) if previous else {}
        self._settle_terminal(con)
        jid = super()._enqueue_with_connection(
            con, sid, owner, payload, construction_approval=construction_approval
        )
        stored = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        )
        seconds = stored.get("max_seconds")
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds <= 0:
            raise WorkspaceError("Hosted compute admission requires a finite original work budget")
        continuing = bool(
            payload.get("resume")
            or payload.get("intent_resolution")
            or stored.get("recovery_of")
            or construction_approval
        )
        lineage = previous_payload.get("host_lineage_id") if continuing else None
        if continuing and lineage is None:
            raise WorkspaceError("Hosted continuation requires its original compute reservation")
        if lineage:
            row = con.execute(
                "SELECT seconds,owner FROM host_lineages WHERE lineage_id=?", (lineage,)
            ).fetchone()
            spent = con.execute(
                "SELECT coalesce(sum(coalesce(used_seconds,seconds)),0) FROM host_reservations WHERE lineage_id=?",
                (lineage,),
            ).fetchone()[0]
            if row is None or row[1] != owner or row[0] <= spent:
                raise WorkspaceError("Original hosted compute allowance is exhausted")
            seconds = min(seconds, row[0] - spent)
            stored["max_seconds"] = seconds
        try:
            reserve(
                con,
                owner,
                jid,
                math.ceil(seconds),
                active_limit=self.host_config.owner_active_limit,
                lineage_id=lineage,
            )
        except HostingError as exc:
            raise WorkspaceError(str(exc)) from exc
        stored["host_lineage_id"] = lineage or jid
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(stored), jid))
        return jid

    def _create_with_connection(
        self,
        con: sqlite3.Connection,
        owner: str,
        prompt: str,
        contract: dict[str, Any] | None,
        authoring_mode: str | None = None,
    ) -> dict[str, Any]:
        if re.fullmatch(r"principal:v1:[a-f0-9]{64}", owner) is None:
            raise WorkspaceError("Private host requires a validated canonical owner")
        return super()._create_with_connection(con, owner, prompt, contract, authoring_mode)

    @contextlib.contextmanager
    def active_job(self, jid: str) -> Iterator[None]:
        self._active.job_id = jid
        try:
            yield
        finally:
            self._active.job_id = None

    def hold_uncertain(self, reason: str | RunnerUncertainError) -> None:
        """Called at the broker boundary before general agent error handling."""
        operation_id = reason.operation_id if isinstance(reason, RunnerUncertainError) else ""
        request_hash = reason.request_hash if isinstance(reason, RunnerUncertainError) else ""
        reason = str(reason)
        if not operation_id:
            identifiers = re.findall(r"(?<![a-f0-9])[a-f0-9]{32}(?![a-f0-9])", reason)
            operation_id = identifiers[-1] if identifiers else ""
        hold_id = hashlib.sha256(reason.encode()).hexdigest()
        jid = getattr(self._active, "job_id", None)
        with super().connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT OR IGNORE INTO host_runner_holds VALUES (?,?)",
                (hold_id, reason[:500]),
            )
            if jid is None:
                # An owner-scoped direct observer has no queued job. Block all
                # current reservations conservatively until runner reconciliation.
                jobs = con.execute(
                    "SELECT job_id FROM host_reservations WHERE used_seconds IS NULL"
                ).fetchall()
            else:
                jobs = [(jid,)]
            for row in jobs:
                con.execute("INSERT OR IGNORE INTO host_holds VALUES (?,?)", (row[0], reason[:500]))
            reservations = []
            for row in jobs:
                reservation = con.execute(
                    "SELECT job_id,owner,seconds,used_seconds,lineage_id FROM host_reservations WHERE job_id=?",
                    (row[0],),
                ).fetchone()
                if reservation is not None:
                    reservations.append(dict(reservation))
            evidence = {
                "version": 1,
                "hold_id": hold_id,
                "host": socket.gethostname(),
                "fence": self.host_config.runner_fence,
                "image": self.host_config.image,
                "operation_id": operation_id,
                "request_hash": request_hash,
                "reason": reason[:500],
                "reservations": reservations,
            }
            existing = con.execute(
                "SELECT document FROM host_runner_hold_evidence WHERE hold_id=?", (hold_id,)
            ).fetchone()
            if existing is None:
                con.execute(
                    "INSERT INTO host_runner_hold_evidence VALUES (?,?)",
                    (hold_id, json.dumps(evidence, sort_keys=True)),
                )

    def clear_runner_hold(self, runner_receipt: Path) -> dict[str, Any]:
        """Offline operator action after the dedicated runner proves terminal cleanup.

        Unknown consumption keeps the original compute reservation fully consumed.
        The operator supplies the actual runner's receipt, never an HTTP body.
        """
        if (
            runner_receipt.is_symlink()
            or not runner_receipt.is_file()
            or runner_receipt.stat().st_size > 4 * 1024 * 1024
        ):
            raise HostingError("A bounded operator-owned runner receipt is required")
        native = json.loads(runner_receipt.read_text())
        operation_id = native.get("operation_id", "")
        if (
            not isinstance(operation_id, str)
            or re.fullmatch(r"[a-f0-9]{32}", operation_id) is None
            or native.get("version") != 1
            or native.get("image") != self.host_config.image
            or native.get("fence") != self.host_config.runner_fence
            or native.get("container_name") != "cadkitty-" + operation_id
            or native.get("state") not in {"done", "interrupted", "error", "cancelled", "rejected"}
        ):
            raise HostingError("Runner receipt does not prove exact terminal launch authority")
        receipt = {
            "version": 1,
            "operation_id": operation_id,
            "runner_receipt_sha256": hashlib.sha256(runner_receipt.read_bytes()).hexdigest(),
            "unknown_consumption": "original_reservation_consumed",
            "automatic_retry": False,
        }
        with HostLock(self.root):
            path = self.root / (".runner-reconciliation-" + operation_id + ".json")
            if path.exists() and json.loads(path.read_text()) != receipt:
                raise HostingError("A different runner reconciliation receipt already exists")
            with self.connect() as con:
                con.execute("BEGIN IMMEDIATE")
                holds = [
                    row
                    for row in con.execute("SELECT hold_id,reason FROM host_runner_holds")
                    if operation_id in re.findall(r"(?<![a-f0-9])[a-f0-9]{32}(?![a-f0-9])", row[1])
                ]
                if not holds and not path.exists():
                    raise HostingError("Runner receipt does not match a durable uncertainty hold")
                for hold in holds:
                    original = con.execute(
                        "SELECT document FROM host_runner_hold_evidence WHERE hold_id=?", (hold[0],)
                    ).fetchone()
                    original_evidence = json.loads(original[0]) if original else None
                    if (
                        original_evidence is None
                        or original_evidence.get("fence") != native.get("fence")
                        or original_evidence.get("image") != native.get("image")
                        or original_evidence.get("operation_id") != operation_id
                        or (
                            original_evidence.get("request_hash")
                            and original_evidence["request_hash"] != native.get("request_hash")
                        )
                    ):
                        raise HostingError(
                            "Terminal runner receipt differs from the original held authority"
                        )
                    if native.get("state") == "rejected" and (
                        native.get("accepted") is not False
                        or original_evidence is None
                        or not original_evidence.get("request_hash")
                        or native.get("request_hash") != original_evidence["request_hash"]
                    ):
                        raise HostingError(
                            "Nonacceptance receipt differs from the original uncertain request"
                        )
                    jobs = con.execute(
                        "SELECT r.job_id,r.owner,r.seconds FROM host_reservations r JOIN host_holds h ON h.job_id=r.job_id WHERE h.reason=? AND r.used_seconds IS NULL",
                        (hold[1],),
                    ).fetchall()
                    for job in jobs:
                        settle(con, job[1], job[0], job[2])
                    con.execute("DELETE FROM host_holds WHERE reason=?", (hold[1],))
                    con.execute("DELETE FROM host_runner_holds WHERE hold_id=?", (hold[0],))
                if not path.exists():
                    write_json(path, receipt)
            return receipt

    def apply_fenced_recovery(
        self,
        *,
        old_host: str,
        job_ids: list[str],
        fencing_evidence: Path,
        restore_id: str | None = None,
        failpoint: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        def checkpoint(name: str) -> None:
            if failpoint is not None:
                failpoint(name)

        with HostLock(self.root):
            marker = self.root / ".restore-required.json"
            identity = current_restore(self.root, restore_id)
            receipt_path = self.root / ".restore-history" / identity["restore_id"] / "recovery.json"
            previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            if marker.exists():
                plan = _fenced_plan_locked(
                    self.root, old_host=old_host, job_ids=job_ids, fencing_evidence=fencing_evidence
                )
                receipt = {**plan, "applied": True, "automatic_retry": False}
                if previous is not None and previous != receipt:
                    raise HostingError("A different fenced recovery receipt already exists")
            else:
                if (
                    previous is None
                    or previous.get("restore_id") != identity["restore_id"]
                    or previous.get("old_host") != old_host
                    or previous.get("job_ids") != sorted(job_ids)
                    or previous.get("fencing_evidence_sha256")
                    != hashlib.sha256(fencing_evidence.read_bytes()).hexdigest()
                ):
                    raise HostingError("No matching completed fenced recovery exists")
                receipt = previous
            with self.connect() as con:
                con.execute("BEGIN IMMEDIATE")
                current = con.execute("SELECT id,state,worker_host FROM jobs").fetchall()
                selected = [row for row in current if row[0] in set(job_ids)]
                if (
                    len(job_ids) != len(set(job_ids))
                    or {row[0] for row in selected} != set(job_ids)
                    or any(row[2] != old_host for row in selected)
                    or any(row[1] == "running" and row[0] not in set(job_ids) for row in current)
                ):
                    raise HostingError("Fenced recovery changed before application")
                running = {row[0] for row in selected if row[1] == "running"}
                if running:
                    if running != set(job_ids):
                        raise HostingError(
                            "Partial fenced recovery is not an atomic original transition"
                        )
                    self._recover_interrupted_jobs(
                        con, fenced_jobs=frozenset(job_ids), fenced_host=old_host
                    )
                elif (job_ids and previous is None) or any(
                    row[1] != "interrupted" for row in selected
                ):
                    raise HostingError(
                        "Original recovery jobs are not in their exact completed states"
                    )
                if con.execute("SELECT 1 FROM jobs WHERE state='running'").fetchone():
                    raise HostingError("Fenced recovery left active work unresolved")
                if previous is None:
                    write_json(receipt_path, receipt)
                checkpoint("before_commit")
            checkpoint("after_commit")
            checkpoint("before_marker_remove")
            if marker.exists():
                marker.unlink()
            return receipt

    def dispose_restored_runner_hold(
        self,
        *,
        restore_id: str,
        hold_ids: list[str],
        operation_id: str,
        old_fence: str,
        old_image: str,
        job_ids: list[str],
        fencing_evidence: Path,
        failpoint: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Explicit old-host disposition, separate from a current runner's receipt.

        Preserve original evidence and fully consume unknown reserved work. No
        new-fence receipt is fabricated and no command is automatically retried.
        """
        if (
            re.fullmatch(r"[a-f0-9]{32}", operation_id) is None
            or re.fullmatch(r"[a-f0-9]{32}", old_fence) is None
            or re.fullmatch(r"sha256:[a-f0-9]{64}", old_image) is None
        ):
            raise HostingError(
                "Disposition requires exact original operation, fence and image identities"
            )
        with HostLock(self.root):
            identity = current_restore(self.root, restore_id)
            history = self.root / ".restore-history" / restore_id
            recovery = history / "recovery.json"
            if (
                (self.root / ".restore-required.json").exists()
                or not recovery.is_file()
                or json.loads(recovery.read_text()).get("restore_id") != restore_id
            ):
                raise HostingError("Original fenced checkpoint recovery must complete first")
            if (
                not hold_ids
                or len(hold_ids) != len(set(hold_ids))
                or len(job_ids) != len(set(job_ids))
            ):
                raise HostingError("Disposition requires exact distinct original holds and jobs")
            captured = {row["hold_id"]: row for row in identity.get("runner_holds", [])}
            if any(hold_id not in captured for hold_id in hold_ids):
                raise HostingError("Disposition hold does not belong to this restore")
            originals: list[dict[str, Any]] = []
            reservations: dict[str, dict[str, Any]] = {}
            old_host: str | None = None
            for hold_id in sorted(hold_ids):
                original = captured[hold_id].get("evidence")
                if (
                    not isinstance(original, dict)
                    or original.get("hold_id") != hold_id
                    or original.get("operation_id") != operation_id
                    or original.get("fence") != old_fence
                    or original.get("image") != old_image
                    or not original.get("host")
                ):
                    raise HostingError("Original runner operation, fence or image differs")
                if old_host is not None and old_host != original["host"]:
                    raise HostingError("Disposition cannot combine different original hosts")
                old_host = original["host"]
                originals.append(original)
                for reservation in original.get("reservations", []):
                    job_id = reservation["job_id"]
                    if job_id in reservations and reservations[job_id] != reservation:
                        raise HostingError("Original affected reservation evidence conflicts")
                    reservations[job_id] = reservation
            if set(job_ids) != set(reservations):
                raise HostingError(
                    "Disposition must name every exact original affected reservation"
                )
            if (
                fencing_evidence.is_symlink()
                or not fencing_evidence.is_file()
                or fencing_evidence.stat().st_size > 65536
            ):
                raise HostingError("A bounded explicit old-runner fencing attestation is required")
            attestation = json.loads(fencing_evidence.read_text())
            expected = {
                "restore_id": restore_id,
                "old_host": old_host,
                "operation_id": operation_id,
                "old_fence": old_fence,
                "old_image": old_image,
                "hold_ids": sorted(hold_ids),
                "job_ids": sorted(job_ids),
            }
            if (
                any(attestation.get(key) != value for key, value in expected.items())
                or attestation.get("fenced") is not True
                or not attestation.get("operator")
                or not attestation.get("evidence")
            ):
                raise HostingError(
                    "Old-runner fencing evidence differs from the exact restore disposition"
                )
            receipt = {
                "version": 1,
                **expected,
                "new_fence": self.host_config.runner_fence,
                "restore_manifest_sha256": identity["manifest_sha256"],
                "fencing_evidence_sha256": hashlib.sha256(
                    fencing_evidence.read_bytes()
                ).hexdigest(),
                "original_hold_evidence": originals,
                "unknown_consumption": "original_reservation_consumed",
                "automatic_retry": False,
            }
            disposition_id = hashlib.sha256(
                json.dumps([restore_id, sorted(hold_ids)]).encode()
            ).hexdigest()
            path = history / ("runner-disposition-" + disposition_id + ".json")
            encoded = json.dumps(receipt, sort_keys=True)
            if path.exists() and json.loads(path.read_text()) != receipt:
                raise HostingError("Disposition replay changed its retained evidence")
            with self.connect() as con:
                con.execute("BEGIN IMMEDIATE")
                prior = con.execute(
                    "SELECT document FROM host_hold_dispositions WHERE id=?", (disposition_id,)
                ).fetchone()
                if prior is not None:
                    if prior[0] != encoded:
                        raise HostingError("Disposition replay conflicts with committed authority")
                    return receipt
                for original in originals:
                    hold_id = original["hold_id"]
                    active = con.execute(
                        "SELECT reason FROM host_runner_holds WHERE hold_id=?", (hold_id,)
                    ).fetchone()
                    retained = con.execute(
                        "SELECT document FROM host_runner_hold_evidence WHERE hold_id=?", (hold_id,)
                    ).fetchone()
                    if (
                        active is None
                        or active[0] != original["reason"]
                        or retained is None
                        or json.loads(retained[0]) != original
                    ):
                        raise HostingError(
                            "Restored uncertainty authority changed before disposition"
                        )
                for job_id, original in reservations.items():
                    current = con.execute(
                        "SELECT owner,seconds,used_seconds,lineage_id FROM host_reservations WHERE job_id=?",
                        (job_id,),
                    ).fetchone()
                    if (
                        current is None
                        or current[0] != original["owner"]
                        or current[1] != original["seconds"]
                        or current[3] != original["lineage_id"]
                        or current[2] not in {None, original["seconds"]}
                    ):
                        raise HostingError(
                            "Original unknown reservation changed before disposition"
                        )
                    settle(con, current[0], job_id, current[1])
                for original in originals:
                    con.execute("DELETE FROM host_holds WHERE reason=?", (original["reason"],))
                    con.execute(
                        "DELETE FROM host_runner_holds WHERE hold_id=?", (original["hold_id"],)
                    )
                if not path.exists():
                    write_json(path, receipt)
                con.execute(
                    "INSERT INTO host_hold_dispositions VALUES (?,?)", (disposition_id, encoded)
                )
                if failpoint:
                    failpoint("before_commit")
            if failpoint:
                failpoint("after_commit")
            return receipt
