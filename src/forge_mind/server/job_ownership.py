"""Legacy job ownership is stored outside model-writable job directories."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def claim(root: Path, job: str, owner: str) -> None:
    with sqlite3.connect(root / "job-owners.sqlite3", timeout=10) as con:
        con.execute("CREATE TABLE IF NOT EXISTS owners (job TEXT PRIMARY KEY, owner TEXT NOT NULL)")
        con.execute("INSERT INTO owners VALUES (?, ?)", (job, owner))


def owns(root: Path, job: str, owner: str) -> bool:
    path = root / "job-owners.sqlite3"
    if not path.is_file():
        return False
    with sqlite3.connect(path, timeout=10) as con:
        row = con.execute("SELECT owner FROM owners WHERE job=?", (job,)).fetchone()
    return row is not None and row[0] == owner
