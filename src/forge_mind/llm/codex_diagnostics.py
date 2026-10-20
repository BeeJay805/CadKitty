"""Bounded CLI lifecycle counts; never retain provider payloads or error text."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge_mind.llm.output_contract import OutputContractError

EVENT_TYPES = (
    "thread.started",
    "turn.started",
    "item.started",
    "item.updated",
    "item.completed",
    "turn.completed",
    "turn.failed",
    "error",
)
LOG_LIMIT = 4 * 1024 * 1024
LINE_LIMIT = 64 * 1024
EVENT_LIMIT = 16_384


class CallDiagnostics:
    """Temporary collector. Only its sanitized record leaves the call scope."""

    def __init__(self, model: str) -> None:
        self.started = time.monotonic()
        self.collected = False
        self.primary_exception: BaseException | None = None
        self.record: dict[str, Any] = {
            "call_id": uuid.uuid4().hex,
            "requested_model": model
            if re.fullmatch(r"gpt-[A-Za-z0-9._-]{1,90}", model)
            else "unrecognized",
            "outcome": "input_rejected",
            "elapsed_seconds": 0.0,
            "process_started": False,
            "returncode": None,
            "events_bytes": 0,
            "stderr_bytes": 0,
            "answer_bytes": 0,
            "event_counts": dict.fromkeys((*EVENT_TYPES, "other"), 0),
            "events_parse_truncated": False,
            "diagnostics_complete": True,
            "collection_error": None,
            "cleanup_error": False,
            "output_contract": None,
        }

    def output_correction_allowed(self) -> bool:
        """Only complete, clean, bounded successful CLI output can be corrected."""
        events = self.record["event_counts"]
        return bool(
            self.collected
            and self.record["process_started"]
            and self.record["returncode"] == 0
            and 0 < self.record["answer_bytes"] <= 512_000
            and self.record["diagnostics_complete"]
            and not self.record["cleanup_error"]
            and self.record["outcome"] == "invalid_response"
            and events["turn.completed"] > 0
            and not events["turn.failed"]
            and not events["error"]
            and not events["other"]
        )

    def rejected_output(self, error: OutputContractError) -> OutputContractError:
        """Keep the safe rejection before temporary cleanup can itself fail."""
        self.record["output_contract"] = {**error.feedback, "retryable": error.retryable}
        return error

    def incomplete(self, code: str) -> None:
        self.record["diagnostics_complete"] = False
        if self.record["collection_error"] is None:
            self.record["collection_error"] = code

    @contextmanager
    def scope(self, root: Path) -> Iterator[None]:
        """Collect before temporary cleanup without replacing execution errors."""
        try:
            yield
        except BaseException as exc:
            if self.record["outcome"] == "success":
                self.record["cleanup_error"] = True
            self.primary_exception = exc
            raise
        finally:
            try:
                self.collect(root)
            except Exception:
                self.incomplete("collection_failed")

    def collect(self, root: Path) -> None:
        """Read bounded event summaries before the caller deletes temporary files."""
        self.collected = True
        for name, key in (
            ("events.jsonl", "events_bytes"),
            ("stderr.txt", "stderr_bytes"),
            ("answer.json", "answer_bytes"),
        ):
            try:
                size = (root / name).stat().st_size
                self.record[key] = max(0, size)
            except FileNotFoundError:
                if name != "answer.json" and self.record["process_started"]:
                    self.incomplete("read_failed")
            except OSError:
                self.incomplete("read_failed")
        events = root / "events.jsonl"
        try:
            with events.open("rb") as stream:
                consumed = 0
                for _ in range(EVENT_LIMIT):
                    line = stream.readline(min(LINE_LIMIT + 1, LOG_LIMIT - consumed + 1))
                    if not line:
                        break
                    consumed += len(line)
                    if consumed > LOG_LIMIT or len(line) > LINE_LIMIT:
                        self.record["events_parse_truncated"] = True
                        self.incomplete("parse_limit")
                        break
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            self.incomplete("malformed_events")
                        kind = event.get("type") if isinstance(event, dict) else None
                        key = kind if isinstance(kind, str) and kind in EVENT_TYPES else "other"
                    except (ValueError, RecursionError):
                        key = "other"
                        self.incomplete("malformed_events")
                    self.record["event_counts"][key] += 1
                else:
                    if stream.read(1):
                        self.record["events_parse_truncated"] = True
                        self.incomplete("parse_limit")
        except FileNotFoundError:
            if self.record["process_started"]:
                self.incomplete("read_failed")
        except OSError:
            self.incomplete("read_failed")

    def finish(self) -> dict[str, Any]:
        if self.record["process_started"] and self.record["returncode"] is None:
            self.incomplete("process_state_unknown")
        self.record["elapsed_seconds"] = round(max(0.0, time.monotonic() - self.started), 6)
        return self.record
