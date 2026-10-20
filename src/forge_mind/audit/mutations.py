"""Defect-injection operators over authored build123d scripts (ADR-0083).

Each operator proposes CANDIDATE mutants for one defect class as pure code transforms --
no geometry here. The audit runner builds each candidate in the sandbox and a CONFIRMER
(confirmers.py) verifies the defect is physically present in the built mesh; candidates
whose defect did not materialize are discarded as invalid trials. That division means
operators optimize for YIELD with honest descriptions, while label correctness is owned
by the confirmers -- a no-op mutation costs a trial, never poisons a rate.

Precision conventions follow attribution.py (PR #124): label->line matching uses the same
token machinery, and a label whose tokens light up too many lines is dropped as ambiguous
rather than guessed at.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from forge_mind.agentic.attribution import _line_words, _tokens
from forge_mind.agentic.params import ParamSpec, extract_params, pinned_float_names

# Token vocabularies for param semantics. Deliberately conservative: a param that does
# not clearly read as a wall / position / envelope dimension is left alone. W4.3 widened
# both thin-class vocabularies (ADR-0012 E0: n>=30 per critical class) -- safe because the
# CONFIRMER owns label correctness: a mis-picked param is a discarded trial, never a
# poisoned rate; wider tokens only cost yield.
_WALL_TOKENS = frozenset(
    {"wall", "thickness", "thick", "thk", "web", "rib", "floor", "shell", "plate", "panel"}
)
_POSITION_TOKENS = frozenset(
    {
        "spacing",
        "pitch",
        "bcd",
        "pcd",
        "centers",
        "offset",
        "cc",
        "span",
        "circle",
        "pos",
        "position",
        "dist",
        "distance",
        "sep",
        "separation",
        "pattern",
        "apart",
    }
)
_ENVELOPE_TOKENS = frozenset(
    {"length", "width", "height", "od", "dia", "diameter", "outer", "overall", "len"}
)
_BORE_SIZE_TOKENS = frozenset({"bore", "hole", "dia", "diameter", "clearance"})
# kwargs that set a cut's axial extent in build123d idioms (Hole(depth=),
# Cylinder(height=), extrude(amount=)).
_DEPTH_KWARGS = ("depth", "height", "h", "amount")

# Ambiguity guard, same spirit as attribution.py: tokens lighting up more lines than this
# are generic for the script and unusable as a locator.
_MAX_LABEL_LINES = 4


@dataclass(frozen=True)
class MutantCandidate:
    """One proposed defect injection: mutated source + what the confirmer must verify."""

    defect_class: str  # blind_where_through | off_brief_stated_dim | sub_floor_wall
    #                  | dropped_named_feature | off_footprint_bore | envelope_violation
    description: str  # human-readable: site + old -> new
    code: str  # the full mutated script
    meta: dict[str, float | int | str] = field(default_factory=dict)


def _param_tokens(name: str) -> set[str]:
    return {w.lower() for w in name.split("_") if len(w) >= 2}


def _tok_match(words: set[str], vocab: frozenset[str] | set[str]) -> bool:
    """Exact or prefix-tolerant token match ('bolt' ~ 'bolts', 'wid' ~ 'width').
    Prefixing needs >=3 chars on both sides so short tokens ('cc') stay exact-only."""
    for w in words:
        for v in vocab:
            if w == v:
                return True
            if len(w) >= 3 and len(v) >= 3 and (w.startswith(v) or v.startswith(w)):
                return True
    return False


def _splice_params(code: str, spec: ParamSpec, new_values: dict[str, float | int]) -> str:
    """override_params without the int-freeze: the production optimizer must never touch
    structure counts (ADR-0073), but the audit's whole point is injecting structural
    defects. Same exact-source-segment splice."""
    merged: dict[str, float | int] = {}
    for k, v in spec.values.items():
        merged[k] = new_values.get(k, v)
    body = ", ".join(f"{k!r}: {v!r}" for k, v in merged.items())
    return code[: spec.start] + "{" + body + "}" + code[spec.end :]


# ------------------------------------------------------------ stated dims ---


def stated_dim_mutants(code: str, stated: frozenset[float]) -> list[MutantCandidate]:
    """Class off_brief_stated_dim: move a brief-stated dimension well past the ADR-0080
    pin window (max(2%, 0.2mm)). Factors chosen to keep the part plausible but wrong."""
    spec = extract_params(code)
    if spec is None:
        return []
    out: list[MutantCandidate] = []
    for name in pinned_float_names(spec, stated):
        old = float(spec.values[name])
        for factor in (1.12, 0.85):
            new = round(old * factor, 4)
            out.append(
                MutantCandidate(
                    defect_class="off_brief_stated_dim",
                    description=f"PARAMS[{name!r}] {old} -> {new} (x{factor})",
                    code=_splice_params(code, spec, {name: new}),
                    meta={"param": name, "old": old, "new": new},
                )
            )
    return out


def undersized_clearance_bore_mutants(
    code: str, target_diameters_mm: list[float]
) -> list[MutantCandidate]:
    """Class undersized_clearance_bore: keep plan association but violate fit."""
    spec = extract_params(code)
    if spec is None:
        return []
    out: list[MutantCandidate] = []
    for name in spec.float_names:
        old = float(spec.values[name])
        if not _tok_match(_param_tokens(name), _BORE_SIZE_TOKENS):
            continue
        if not any(abs(old - target) <= 0.25 for target in target_diameters_mm):
            continue
        new = round(old - 0.5, 4)
        if new <= 0:
            continue
        out.append(
            MutantCandidate(
                defect_class="undersized_clearance_bore",
                description=f"PARAMS[{name!r}] {old} -> {new} (inside assignment window)",
                code=_splice_params(code, spec, {name: new}),
                meta={"param": name, "old": old, "new": new},
            )
        )
    return out


# --------------------------------------------------------- sub-floor walls ---


# Params the thinnest-float fallback must never touch: hole/rounding semantics are
# routinely wall-SCALED (a 3-8 mm bore) but thinning them ADDS material or trims an
# edge -- never a sub-floor wall. Token-matched sites are exempt (an explicit
# "wall"/"thk" name outranks these).
_NEVER_WALL_TOKENS = frozenset(
    {"bore", "dia", "diameter", "hole", "radius", "rad", "clearance", "chamfer", "fillet", "gap"}
)


# Params that DECLARE a limit rather than size a wall: mutating `wall_min` rewrites the
# spec the validator is told to enforce, so the part "passes" its own mutated contract --
# a spec-drift, not a geometry defect. Measured live (2026-07-10 audit): the one escape in
# 35 valid mutants was exactly this site. Declarations are never mutation sites.
_DECLARATION_TOKENS = frozenset({"min", "max", "limit", "budget", "tol", "tolerance"})


def _wall_semantic(name: str) -> bool:
    """Wall tokens, plus the `t` / `_t` suffix idiom (`base_t`, `wall_t`, bare `t`) that
    token matching cannot see because single-letter words are dropped by `_param_tokens`."""
    low = name.lower()
    toks = _param_tokens(name)
    if _tok_match(toks, _DECLARATION_TOKENS):
        return False
    if low == "t" or low.endswith("_t"):
        return True
    return _tok_match(toks, _WALL_TOKENS)


def subfloor_wall_mutants(code: str, min_wall_mm: float | None) -> list[MutantCandidate]:
    """Class sub_floor_wall: thin a wall-semantic param below the declared minimum.

    Two strategies (W4.3, ADR-0012 E0): token/suffix-matched params first; when NO param
    reads as a wall, fall back to scaling the THINNEST wall-scaled float. The fallback is
    honest by construction -- the confirmer measures the built mesh's thickness percentile,
    so a fallback that thinned a non-wall (a chamfer, a clearance) simply fails to confirm
    and is discarded as an invalid trial."""
    floor = min_wall_mm if min_wall_mm is not None else 1.2  # generic FDM floor fallback
    spec = extract_params(code)
    if spec is None:
        return []
    out: list[MutantCandidate] = []

    def _emit(name: str, old: float, kind: str) -> None:
        new = round(floor * 0.55, 4)
        if new >= old:
            return
        out.append(
            MutantCandidate(
                defect_class="sub_floor_wall",
                description=f"PARAMS[{name!r}] {old} -> {new} (below min wall {floor}, {kind})",
                code=_splice_params(code, spec, {name: new}),
                meta={"param": name, "old": old, "new": new, "floor": floor, "site": kind},
            )
        )

    # wall-scale window: plausibly a wall only if wall-scaled (a 60mm "wall_length" is
    # not a wall) and not already sub-floor (a known-good's governing wall sits at/above)
    in_scale = [
        (name, float(spec.values[name]))
        for name in spec.float_names
        if floor * 0.9 <= float(spec.values[name]) <= floor * 4.0
    ]
    matched = [(n, v) for n, v in in_scale if _wall_semantic(n)]
    for name, old in matched:
        _emit(name, old, "wall-token")
    if not matched:
        # scale-thinnest fallback: the thinnest in-scale float is the likeliest governing
        # wall in scripts that name nothing wall-like -- but never a hole/rounding param
        # (wall-scaled yet thinning one adds material; the confirmer would discard it).
        pool = [
            (n, v)
            for n, v in in_scale
            if not _tok_match(_param_tokens(n), _NEVER_WALL_TOKENS)
            and not _tok_match(_param_tokens(n), _DECLARATION_TOKENS)
        ]
        if pool:
            name, old = min(pool, key=lambda kv: kv[1])
            _emit(name, old, "thinnest-float fallback")
    return out


# ------------------------------------------------------ dropped named features ---


def _label_statements(tree: ast.Module, code: str, label: str) -> list[ast.stmt]:
    """Self-contained statements whose source mentions the label's distinctive tokens.
    Searches module body and one level of With/For nesting (the BuildPart idiom)."""
    toks = _tokens(label)
    if not toks:
        return []
    hits: list[ast.stmt] = []

    def _scan(body: list[ast.stmt]) -> None:
        for stmt in body:
            # imports and the PARAMS declaration are not feature-BUILD code -- matching them (the
            # PARAMS dict literally contains "bolt_circle_r"/"n_bolts") made deletion try to remove
            # PARAMS and NameError the whole script. They are never the cut to delete.
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PARAMS" for t in stmt.targets
            ):
                continue
            seg = ast.get_source_segment(code, stmt) or ""
            words: set[str] = set()
            for line in seg.splitlines():
                words |= _line_words(line)
            if toks & words:
                # prefer the SMALLEST enclosing statement: recurse before recording a
                # whole `with` block when an inner statement carries the tokens
                inner: list[ast.stmt] = []
                if isinstance(stmt, (ast.With, ast.For)):
                    _scan(stmt.body)
                    inner = [h for h in hits if h in stmt.body]
                if not inner:
                    hits.append(stmt)

    _scan(tree.body)
    return hits


class _StatementDropper(ast.NodeTransformer):
    """Remove the doomed statements (by node id) from module / with / for bodies."""

    def __init__(self, doomed: set[int]) -> None:
        self._doomed = doomed

    def _prune(self, body: list[ast.stmt]) -> list[ast.stmt]:
        kept = [s for s in body if id(s) not in self._doomed]
        return kept or [ast.Pass()]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._prune(node.body)
        return node

    def visit_With(self, node: ast.With) -> ast.With:
        self.generic_visit(node)
        node.body = self._prune(node.body)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.body = self._prune(node.body)
        return node


def _deletion_is_safe(tree: ast.Module, stmts: list[ast.stmt]) -> bool:
    """True when deleting ``stmts`` cannot NameError the rest of the script: no name they BIND is
    LOADed anywhere outside them. A pure side-effect statement (a ``Hole(...)`` cut, a
    ``with Locations(...): ...`` block) binds nothing and is always safe; a ``profile = loft(...)``
    that a later line consumes is not. This is exactly why v1's deletion broke the build ~20/22 --
    it deleted a binding a downstream statement referenced."""
    excluded: set[int] = set()
    bound: set[str] = set()
    for s in stmts:
        for n in ast.walk(s):
            excluded.add(id(n))
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
            elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
                bound.add(n.optional_vars.id)
    if not bound:
        return True
    used_outside = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in excluded
    }
    return not (bound & used_outside)


def dropped_feature_mutants(code: str, feature_labels: list[str]) -> list[MutantCandidate]:
    """Class dropped_named_feature: remove the code that builds a brief-named feature.

    Two strategies, both emitted as separate candidates when applicable (confirmers pick
    the survivors): (a) zero an INT count param whose name matches the label; (b) delete
    the matched statement(s) -- but only when DATAFLOW-SAFE (they bind no name used later),
    the v2.1 guard that stops the ~20/22 build breakage v1 measured."""
    out: list[MutantCandidate] = []
    spec = extract_params(code)
    try:
        ast.parse(code)
    except SyntaxError:
        return []

    for label in feature_labels:
        toks = _tokens(label)
        if not toks:
            continue
        # (a) count-param zeroing
        if spec is not None:
            for name, value in spec.values.items():
                if isinstance(value, int) and value >= 2 and _tok_match(_param_tokens(name), toks):
                    out.append(
                        MutantCandidate(
                            defect_class="dropped_named_feature",
                            description=f"'{label}': PARAMS[{name!r}] {value} -> 0",
                            code=_splice_params(code, spec, {name: 0}),
                            meta={"label": label, "param": name, "old": value},
                        )
                    )
        # (b) statement deletion: scan and prune the SAME tree (id-based pruning), but ONLY when
        # the deletion is dataflow-safe (binds no name a later line uses) -- else it NameErrors.
        tree = ast.parse(code)
        stmts = _label_statements(tree, code, label)
        if stmts and len(stmts) <= _MAX_LABEL_LINES and _deletion_is_safe(tree, stmts):
            try:
                pruned = _StatementDropper(set(map(id, stmts))).visit(tree)
                new_code = ast.unparse(ast.fix_missing_locations(pruned))
            except Exception:
                continue
            out.append(
                MutantCandidate(
                    defect_class="dropped_named_feature",
                    description=f"'{label}': deleted {len(stmts)} statement(s) building it",
                    code=new_code,
                    meta={"label": label, "dropped_statements": len(stmts)},
                )
            )
    return out


# -------------------------------------------------------- blind-where-through ---


def blind_hole_mutants(
    code: str, hole_labels: list[str], depth_hint_mm: float | None = None
) -> list[MutantCandidate]:
    """Class blind_where_through: make a declared through hole stop short of piercing.
    Text-level and line-scoped, two idioms on label-matched lines:

      * an explicit axial kwarg (`depth=` / `height=` / `amount=`) is halved:
        `depth=EXPR` -> `depth=(EXPR)*0.5`;
      * a bare through-`Hole(...)` (no depth kwarg = build123d through-all) gains an
        injected `depth=depth_hint_mm` -- the hint comes from the BASELINE geometry
        (the runner passes ~0.45x the part's smallest extent), so the cut plausibly
        stops inside the stock. No hint -> that idiom is skipped, never guessed.

    Every candidate must re-parse; the confirmer then proves blindness on the mesh."""
    lines = code.splitlines(keepends=False)
    out: list[MutantCandidate] = []
    kw_re = re.compile(r"\b(" + "|".join(_DEPTH_KWARGS) + r")\s*=\s*([^,)]+)")

    def _mutable(ln: str) -> bool:
        return bool(kw_re.search(ln)) or (
            depth_hint_mm is not None and "Hole(" in ln and ln.rstrip().endswith(")")
        )

    for label in hole_labels:
        toks = _tokens(label)
        if not toks:
            continue
        # Ambiguity is judged on MUTABLE matches (lines carrying a cut idiom), not raw
        # label mentions -- a label echoed in 14 comments/param lines but cutting on 2
        # is precisely located for mutation purposes.
        hit_ix = [i for i, ln in enumerate(lines) if (toks & _line_words(ln)) and _mutable(ln)]
        if not hit_ix or len(hit_ix) > _MAX_LABEL_LINES:
            continue
        for i in hit_ix:
            ln = lines[i]
            new_ln: str | None = None
            desc = ""
            m = kw_re.search(ln)
            if m is not None:
                kw, expr = m.group(1), m.group(2).strip()
                if depth_hint_mm is not None:
                    # Halving is not enough: authors over-specify cut depth (a 15 mm cut through a
                    # 6 mm wall halves to 7.5 mm and STILL pierces -- measured, 4/4 v2 discards).
                    # The geometry-aware hint (< the part's smallest extent) is shorter than ANY
                    # material the bore crosses, so the cut reliably stops short = blind.
                    if kw == "amount":
                        # extrude amount is SIGNED: keep the direction, shrink the magnitude
                        repl = f"{kw}=({depth_hint_mm:.3f}) * (1.0 if ({expr}) >= 0 else -1.0)"
                    else:
                        repl = f"{kw}={depth_hint_mm:.3f}"
                    new_ln = ln[: m.start()] + repl + ln[m.end() :]
                    desc = f"'{label}' line {i + 1}: {kw} -> {depth_hint_mm:.2f} (geometry-aware blind)"
                else:
                    new_ln = ln[: m.start()] + f"{kw}=({expr})*0.5" + ln[m.end() :]
                    desc = f"'{label}' line {i + 1}: {kw}=({expr})*0.5 (no hint, best-effort)"
            elif depth_hint_mm is not None and "Hole(" in ln and ln.rstrip().endswith(")"):
                j = ln.rfind(")")
                new_ln = ln[:j] + f", depth={depth_hint_mm:.3f}" + ln[j:]
                desc = f"'{label}' line {i + 1}: bare Hole gains depth={depth_hint_mm:.3f}"
            if new_ln is None:
                continue
            mutated = "\n".join([*lines[:i], new_ln, *lines[i + 1 :]])
            try:
                ast.parse(mutated)
            except SyntaxError:
                continue  # a malformed splice never ships as a candidate
            out.append(
                MutantCandidate(
                    defect_class="blind_where_through",
                    description=desc,
                    code=mutated,
                    meta={"label": label, "line": i + 1},
                )
            )
            break  # one site per label
    return out


# --------------------------------------------------------- off-footprint bore ---


def off_footprint_mutants(code: str) -> list[MutantCandidate]:
    """Class off_footprint_bore: scale a position-semantic param so bores wander off
    their pattern while size/count stay right. The confirmer diffs measured bore
    centroids baseline-vs-mutant, so no stated anchor is required."""
    spec = extract_params(code)
    if spec is None:
        return []
    out: list[MutantCandidate] = []
    for name in spec.float_names:
        if not _tok_match(_param_tokens(name), _POSITION_TOKENS):
            continue
        old = float(spec.values[name])
        if old <= 1.0:  # a sub-mm "offset" is not a bore pattern
            continue
        new = round(old * 1.3, 4)
        out.append(
            MutantCandidate(
                defect_class="off_footprint_bore",
                description=f"PARAMS[{name!r}] {old} -> {new} (x1.3 position drift)",
                code=_splice_params(code, spec, {name: new}),
                meta={"param": name, "old": old, "new": new},
            )
        )
    return out


# --------------------------------------------------------- envelope violation ---


def envelope_mutants(code: str) -> list[MutantCandidate]:
    """Class envelope_violation: scale the dominant envelope-semantic dimension x1.4.
    Falls back to the largest float param when nothing carries envelope tokens (the
    description says which, and the confirmer checks the measured bbox regardless)."""
    spec = extract_params(code)
    if spec is None:
        return []
    named = [
        (name, float(spec.values[name]))
        for name in spec.float_names
        if _tok_match(_param_tokens(name), _ENVELOPE_TOKENS)
    ]
    pool = named or [
        (name, float(spec.values[name]))
        for name in spec.float_names
        if float(spec.values[name]) >= 10.0
    ]
    if not pool:
        return []
    name, old = max(pool, key=lambda kv: kv[1])
    new = round(old * 1.4, 4)
    kind = "envelope-token" if named else "largest-float fallback"
    return [
        MutantCandidate(
            defect_class="envelope_violation",
            description=f"PARAMS[{name!r}] {old} -> {new} (x1.4, {kind})",
            code=_splice_params(code, spec, {name: new}),
            meta={"param": name, "old": old, "new": new},
        )
    ]


__all__ = [
    "MutantCandidate",
    "blind_hole_mutants",
    "dropped_feature_mutants",
    "envelope_mutants",
    "off_footprint_mutants",
    "stated_dim_mutants",
    "subfloor_wall_mutants",
]
