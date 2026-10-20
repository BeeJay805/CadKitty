"""Tamper-proof acceptance checker -- the honest exit condition of the north-star loop.

The loop's success signal MUST be un-gameable: the cheapest way to make a test pass is to
weaken the test, and on a no-false-PASS engine that silently destroys the moat. So the
acceptance spec for each north star is HASH-PINNED (north_stars.lock.json). ``check``
refuses to declare PASS unless the spec's hash still matches the pinned lock -- if the
loop edited a condition to go green, the hash diverges and the checker reports TAMPERED,
never PASS. Only a HUMAN deliberately running ``lock`` re-blesses a spec change.

Usage:
  acceptance.py check <north_star> <run_dir>   -> evaluate the run's report vs the frozen
                                                  spec; exit 0 iff a verified PASS.
  acceptance.py verify <north_star>            -> spec hash still matches the lock? exit 0/1.
  acceptance.py lock [<north_star>]            -> (HUMAN ONLY) pin the current spec hash(es).
  acceptance.py guard-diff <base_ref>          -> list sensitive files the working tree
                                                  changed vs base_ref; exit 3 if any (the
                                                  loop must STOP for human review).

The registry (north_stars.json) is a list of:
  {name, status, prompt, accept: [ {path, op, value, desc}, ... ]}
``path`` is dotted into functional_report.json; ``op`` in
{eq, ne, gte, lte, in, true, false, present, all_true}.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "north_stars.json"
LOCK = HERE / "north_stars.lock.json"


def _repo_root() -> Path:
    """The git repo root, found by walking up for .git -- robust to where this file lives
    (it moved from .claude/skills/ to scripts/, and a hardcoded parent index silently ran
    git OUTSIDE the repo, which fail-closed every guard). Never hardcode the depth."""
    p = HERE
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return HERE.parents[1]


# Paths the LOOP may never edit to achieve a pass -- the spec + this checker itself.
# HARD STOP (exit 3), human-only, no exceptions: a loop that can edit its own exit
# condition is worthless.
SENSITIVE_PREFIXES = ("scripts/north_star_loop/",)

# Validator-tier paths (founder 2026-07-12: "you are allowed to touch validator if it is
# the right decision and doesn't make it easier to pass"): the loop MAY change these under
# the STRICTNESS RATCHET (SKILL.md) -- coverage/strictness only, never easier to pass,
# two-sided + seeded-defect tests, adversarial not-easier-to-pass review. guard-diff
# REPORTS them (exit 4) so every iteration surfaces validator edits explicitly instead of
# hard-stopping; the burden of proof rides with the fix, never silently.
RATCHET_PREFIXES = (
    "src/forge_mind/validator/",
    "src/forge_mind/agentic/expert_review.py",
    "src/forge_mind/assembly/fits.py",
)


def _load_registry() -> dict[str, dict[str, Any]]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {ns["name"]: ns for ns in data}


def _spec_hash(accept: list[dict[str, Any]]) -> str:
    canon = json.dumps(accept, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _dig(report: dict[str, Any], path: str) -> Any:
    cur: Any = report
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur


def _eval(cond: dict[str, Any], report: dict[str, Any]) -> tuple[bool, str]:
    actual = _dig(report, cond["path"])
    op = cond["op"]
    want = cond.get("value")
    ok: bool
    if op == "eq":
        ok = actual == want
    elif op == "ne":
        ok = actual != want
    elif op == "gte":
        ok = isinstance(actual, int | float) and actual >= want
    elif op == "lte":
        ok = isinstance(actual, int | float) and actual <= want
    elif op == "in":
        ok = actual in want
    elif op == "true":
        ok = bool(actual) is True
    elif op == "false":
        ok = bool(actual) is False
    elif op == "present":
        ok = actual is not None
    elif op == "all_true":
        ok = isinstance(actual, list) and len(actual) > 0 and all(bool(x) for x in actual)
    else:
        return False, f"unknown op {op!r}"
    return ok, f"{'ok ' if ok else 'NO '} {cond.get('desc', cond['path'])}: got {actual!r}"


def cmd_lock(name: str | None) -> int:
    reg = _load_registry()
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    names = [name] if name else list(reg)
    for n in names:
        lock[n] = _spec_hash(reg[n]["accept"])
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"locked: {', '.join(names)}")
    return 0


def cmd_verify(name: str) -> int:
    reg = _load_registry()
    if not LOCK.exists():
        print("TAMPERED: no lock file -- run `lock` (human) to bless the specs first")
        return 1
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cur = _spec_hash(reg[name]["accept"])
    if lock.get(name) != cur:
        print(f"TAMPERED: {name} spec hash {cur} != pinned {lock.get(name)}")
        return 1
    print(f"intact: {name} ({cur})")
    return 0


def cmd_check(name: str, run_dir: str) -> int:
    if cmd_verify(name) != 0:
        print("REFUSING to check: the frozen spec was modified. A pass here is not valid.")
        return 2
    report_path = Path(run_dir) / "functional_report.json"
    if not report_path.exists():
        print(f"FAIL: no functional_report.json in {run_dir}")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reg = _load_registry()
    results = [_eval(c, report) for c in reg[name]["accept"]]
    for _ok, msg in results:
        print(f"  {msg}")
    passed = all(ok for ok, _ in results)
    print(f"{'PASS' if passed else 'FAIL'}: {name} ({sum(ok for ok, _ in results)}/{len(results)})")
    return 0 if passed else 1


def cmd_smell(base_ref: str, name: str) -> int:
    """Deterministic OVERFIT smell-check (founder 2026-07-12: 'not just a narrow fix to the
    north star'). A general capability fix does NOT need the benchmark's name or signature
    numbers in engine SOURCE -- if the loop's diff drops a north-star signature token into
    src/ (not tests -- tests SHOULD name the benchmark), that is special-casing the
    instance, not fixing the class. Exit 3 -> STOP for human review. Fail-closed."""
    reg = _load_registry()
    tokens = [t.lower() for t in reg.get(name, {}).get("signature_tokens", [])]
    if not tokens:
        print(f"smell: no signature_tokens for {name!r} -- cannot check (add them). Fail-closed.")
        return 3
    try:
        diff = subprocess.run(
            ["git", "diff", base_ref, "--", "src/forge_mind"],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            check=True,
        ).stdout
    except Exception as exc:
        print(f"smell could not run ({exc}); treat as SUSPECT (fail-closed)")
        return 3
    added = [
        ln[1:].lower()
        for ln in diff.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    hits = sorted({t for ln in added for t in tokens if t in ln})
    if hits:
        print("OVERFIT SMELL -- benchmark signature token(s) added to engine SOURCE:")
        for h in hits:
            print(f"  '{h}' -- the fix may special-case the north star, not the class. STOP.")
        return 3
    print(f"clean: no {name} signature tokens leaked into src/ (fix is not obviously narrow)")
    return 0


def cmd_baselines() -> int:
    """List the north stars the loop must PROTECT from regression: any whose status marks
    it passing. After every accepted fix these must STILL pass their unchanged spec (the
    loop re-runs + re-checks them); a regression -> revert the fix. Printed one per line."""
    reg = _load_registry()
    protected = [n for n, ns in reg.items() if "passing" in str(ns.get("status", "")).lower()]
    for n in protected:
        print(n)
    return 0


def cmd_metrics(run_dir: str) -> int:
    """Extract a run's EFFICIENCY metrics (report-tier, never part of the correctness
    spec): authoring effort + cost, so an optimization can be proven output-preserving
    (correctness spec unchanged) AND cheaper. Lower total_iterations / cost at an
    identical acceptance verdict = a real efficiency win. Prints JSON."""
    report_path = Path(run_dir) / "functional_report.json"
    if not report_path.exists():
        print(json.dumps({"error": "no report"}))
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parts = report.get("parts", []) or []
    metrics = {
        "disposition": report.get("disposition"),
        "n_parts": len(parts),
        "total_authoring_iterations": sum(int(p.get("iterations", 0) or 0) for p in parts),
        "parts_rebuilt": sum(1 for p in parts if (p.get("stopped_reason") or "") != "accepted"),
        "cost_usd": report.get("total_cost_usd", 0.0),
    }
    print(json.dumps(metrics))
    return 0


def cmd_guard_diff(base_ref: str) -> int:
    root = _repo_root()
    try:
        # committed changes vs the loop's start ref AND uncommitted/untracked working-tree
        # changes -- a re-run uses the working tree, so an unstaged gate edit counts too.
        committed = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            capture_output=True,
            text=True,
            cwd=root,
            check=True,
        ).stdout.splitlines()
        working = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=root,
            check=True,
        ).stdout.splitlines()
    except Exception as exc:
        print(f"guard-diff could not run ({exc}); treat as SENSITIVE (fail-closed)")
        return 3
    files = set(committed) | {line[3:].strip() for line in working if len(line) > 3}
    touched = sorted(f for f in files if any(f.startswith(p) for p in SENSITIVE_PREFIXES))
    if touched:
        print("SENSITIVE paths changed -- the loop must STOP for human review:")
        for t in touched:
            print(f"  {t}")
        return 3
    ratchet = sorted(f for f in files if any(f.startswith(p) for p in RATCHET_PREFIXES))
    if ratchet:
        print(
            "VALIDATOR-TIER paths changed -- the STRICTNESS RATCHET applies (founder "
            "2026-07-12): coverage/strictness only, never easier to pass; two-sided + "
            "seeded-defect tests; adversarial not-easier-to-pass review; spec/lock untouched:"
        )
        for t in ratchet:
            print(f"  {t}")
        return 4
    print("clean: no sensitive (spec/gate) or validator-tier paths changed by the loop")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 64
    cmd = sys.argv[1]
    if cmd == "lock":
        return cmd_lock(sys.argv[2] if len(sys.argv) > 2 else None)
    if cmd == "verify":
        return cmd_verify(sys.argv[2])
    if cmd == "check":
        return cmd_check(sys.argv[2], sys.argv[3])
    if cmd == "guard-diff":
        return cmd_guard_diff(sys.argv[2] if len(sys.argv) > 2 else "HEAD")
    if cmd == "baselines":
        return cmd_baselines()
    if cmd == "metrics":
        return cmd_metrics(sys.argv[2])
    if cmd == "smell":
        return cmd_smell(sys.argv[2], sys.argv[3])
    print(f"unknown command {cmd!r}")
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
