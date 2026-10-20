"""Defect injection (ADR-0119): the unlimited-repair-pairs policy.

Take VALIDATED-GREEN code, break it in the measured failure classes on purpose, keep
only mutants the verifier CONFIRMS are broken the intended way, and write
(green, mutant, violations) triples to a training log. The injectors are deliberately
dumb -- correctness comes from the filter, not the mutation: a mutant survives ONLY if
it (a) still executes and (b) gained at least one new violation (of the targeted kind
when one is declared). That inversion (inject cheap, verify hard) is what makes the
data supply unlimited without ever trusting an unverified label.

Injector map (each targets a measured class from TEST_LOG L62/L66):
  drop_additive_op   -> presence / count_pattern (the absent-boss class)
  grow_subtract      -> min_wall (the thin-wall class: inner bores swell toward walls)
  shrink_base        -> envelope (the wrong-envelope class)
  shift_locations    -> mapped_position / presence (the misplaced-pattern class)

Rows append to ``~/.forge-train/repairs.jsonl`` (override: FORGE_REPAIR_LOG_PATH),
schema-stamped like parts.jsonl: full label_env + injector provenance, so the QLoRA
lane filters by field read, never forensics.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from forge_mind.agentic.reward import RewardVector, reward_vector

if TYPE_CHECKING:
    from forge_mind.agentic.brief import DesignBrief

_ADDITIVE_CALLS = ("extrude", "revolve", "sweep", "loft", "Cylinder", "Box", "Sphere", "Cone")


@dataclass(frozen=True)
class Mutant:
    code: str
    injector: str
    target_kind: str  # the violation kind this injector aims for
    note: str


def _unparse(tree: ast.AST) -> str:
    return ast.unparse(tree)


def _is_subtract_call(node: ast.Call) -> bool:
    return any(
        isinstance(kw.value, ast.Attribute)
        and kw.arg == "mode"
        and getattr(kw.value, "attr", "") == "SUBTRACT"
        for kw in node.keywords
    )


def _additive_expr(stmt: ast.stmt) -> bool:
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    c = stmt.value
    nm = c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", "")
    return nm in _ADDITIVE_CALLS and not _is_subtract_call(c)


def _drop_targets(body: list[ast.stmt], out: list[tuple[list[ast.stmt], int]]) -> None:
    """Deletable sites, recursively. An additive stmt that is its block's ONLY child is
    represented by its enclosing With (drop the whole pattern block, never emit an empty
    block); otherwise the stmt itself."""
    for i, stmt in enumerate(body):
        if _additive_expr(stmt) and len(body) > 1:
            out.append((body, i))
        elif isinstance(stmt, ast.With):
            if len(stmt.body) == 1 and _additive_expr(stmt.body[0]) and len(body) > 1:
                out.append((body, i))
            else:
                _drop_targets(stmt.body, out)
        elif isinstance(stmt, (ast.For, ast.If)):
            _drop_targets(stmt.body, out)


def drop_additive_op(code: str) -> Iterator[Mutant]:
    """Remove ONE additive feature (a statement, or its whole Locations block when the
    feature is the block's only child) -> the absent-feature class."""
    try:
        n_targets = 0
        probe: list[tuple[list[ast.stmt], int]] = []
        _drop_targets(ast.parse(code).body, probe)
        n_targets = len(probe)
    except SyntaxError:
        return
    # skip target 0 (usually the base primitive -- removing it just kills the build)
    for n in range(1, n_targets):
        mutated = ast.parse(code)
        fresh: list[tuple[list[ast.stmt], int]] = []
        _drop_targets(mutated.body, fresh)
        if n >= len(fresh):
            continue
        b, j = fresh[n]
        del b[j]
        yield Mutant(
            _unparse(mutated), "drop_additive_op", "presence", f"dropped additive target {n}"
        )


def grow_subtract_ladder(code: str) -> Iterator[Mutant]:
    """Run-1 measured factor 1.3 minting ZERO rows across 83 bases -- too gentle to
    breach real walls. Climb a factor ladder; the verifier filter keeps only confirmed
    breaks, and the caller dedups by mutant SHA."""
    for factor in (1.4, 1.8, 2.5):
        yield from grow_subtract(code, factor=factor)


def grow_subtract(code: str, factor: float = 1.3) -> Iterator[Mutant]:
    """Scale every SUBTRACT radius/first-arg up -> inner voids swell toward outer walls:
    the thin-wall class."""

    class _Grow(ast.NodeTransformer):
        def __init__(self) -> None:
            self.hits = 0

        def visit_Call(self, node: ast.Call) -> ast.Call:
            self.generic_visit(node)
            if not _is_subtract_call(node):
                return node
            # real code writes radius as a KEYWORD as often as positionally -- scale
            # whichever dimension-carrying slot exists (radius kw first, then arg 0)
            for kw in node.keywords:
                if (
                    kw.arg in ("radius", "r")
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, (int, float))
                ):
                    kw.value = ast.Constant(round(float(kw.value.value) * factor, 3))
                    self.hits += 1
                    return node
            if node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, (int, float)):
                    node.args[0] = ast.Constant(round(float(first.value) * factor, 3))
                    self.hits += 1
            return node

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    grower = _Grow()
    tree = grower.visit(tree)
    if grower.hits:
        ast.fix_missing_locations(tree)
        yield Mutant(
            _unparse(tree), "grow_subtract", "min_wall", f"x{factor} on {grower.hits} subtract dims"
        )


def shrink_base(code: str, factor: float = 0.7) -> Iterator[Mutant]:
    """Scale the FIRST base-primitive call's numeric args -> the envelope class."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    done = False

    class _Shrink(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.Call:
            nonlocal done
            self.generic_visit(node)
            nm = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if not done and nm in ("Box", "Cylinder") and not _is_subtract_call(node):
                changed = False
                for k, arg in enumerate(node.args):
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                        node.args[k] = ast.Constant(round(float(arg.value) * factor, 3))
                        changed = True
                if changed:
                    done = True
            return node

    tree = _Shrink().visit(tree)
    if done:
        ast.fix_missing_locations(tree)
        yield Mutant(_unparse(tree), "shrink_base", "envelope", f"x{factor} on the base primitive")


def shift_locations(code: str, dx: float = 11.0) -> Iterator[Mutant]:
    """Shift the first Locations(...) tuple -> misplaced-pattern class."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    done = False

    class _Shift(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.Call:
            nonlocal done
            self.generic_visit(node)
            nm = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if not done and nm in ("Locations", "GridLocations", "PolarLocations"):
                for k, arg in enumerate(node.args):
                    if isinstance(arg, ast.Tuple) and arg.elts:
                        first = arg.elts[0]
                        if isinstance(first, ast.Constant) and isinstance(
                            first.value, (int, float)
                        ):
                            arg.elts[0] = ast.Constant(round(float(first.value) + dx, 3))
                            done = True
                            break
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                        node.args[k] = ast.Constant(round(float(arg.value) + dx, 3))
                        done = True
                        break
            return node

    tree = _Shift().visit(tree)
    if done:
        ast.fix_missing_locations(tree)
        yield Mutant(
            _unparse(tree), "shift_locations", "mapped_position", f"+{dx}mm on a pattern anchor"
        )


ALL_INJECTORS = (drop_additive_op, grow_subtract_ladder, shrink_base, shift_locations)


def _repair_log_path() -> Path:
    return Path(
        os.environ.get("FORGE_REPAIR_LOG_PATH", "")
        or Path.home() / ".forge-train" / "repairs.jsonl"
    )


def mine_repair_pairs(
    code: str,
    brief: DesignBrief,
    *,
    label: str = "",
    prompt: str | None = None,
    write: bool = True,
    baseline: RewardVector | None = None,
) -> list[dict[str, object]]:
    """Run every injector on validated-green ``code``; keep verifier-confirmed breaks;
    optionally append rows to the repair log. Returns the kept rows.

    The green baseline is re-measured here unless supplied -- a base that is not
    ``passed`` is REFUSED (never mine from an unverified 'green')."""
    base = baseline or reward_vector(code, brief, prompt=prompt)
    if not base.passed:
        return []
    base_kinds = {v.kind for v in base.violations}
    kept: list[dict[str, object]] = []
    for injector in ALL_INJECTORS:
        for mutant in injector(code):
            rv = reward_vector(mutant.code, brief, prompt=prompt)
            if not rv.executable:
                continue
            new = [v for v in rv.violations if v.kind not in base_kinds]
            if not new:
                continue
            on_target = any(v.kind == mutant.target_kind for v in new)
            row: dict[str, object] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "label": label,
                "prompt": prompt or "",
                "green_code": code,
                "green_sha": hashlib.sha256(code.encode()).hexdigest()[:16],
                "mutant_code": mutant.code,
                "mutant_sha": hashlib.sha256(mutant.code.encode()).hexdigest()[:16],
                "injector": mutant.injector,
                "target_kind": mutant.target_kind,
                "on_target": on_target,
                "note": mutant.note,
                "new_violations": [
                    {
                        "kind": v.kind,
                        "message": v.message,
                        "measured": v.measured,
                        "required": v.required,
                        "feature": v.feature,
                    }
                    for v in new
                ],
                "mutant_scalar": rv.scalar,
                "label_env": {
                    "feature_presence": os.environ.get("FORGE_FEATURE_PRESENCE", "") == "1",
                    "vision_model": os.environ.get("FORGE_VISION_MODEL", "qwen2.5vl:7b"),
                },
            }
            kept.append(row)
    if write and kept:
        out = _repair_log_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return kept


__all__ = ["ALL_INJECTORS", "Mutant", "mine_repair_pairs"]
