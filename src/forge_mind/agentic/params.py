"""The parametric contract (ADR-0073 Stage 1): treat an authored script as f(theta) -> mesh.

The measured failure of the generation engine is PARAMETER error, not structure error (the
crankshaft builds recognizably but with wrong proportions/offsets, and three prompt-level levers
measured no gain). The fix is a division of labor: the LLM proposes the STRUCTURE with explicit
named parameters; a numerical optimizer fits the parameters against a deterministic loss
(``param_opt.py``). This module is the contract's mechanical half:

  * the author (with ``FORGE_PARAM_OPT=1``) writes a literal dict at the top of the script::

        PARAMS = {"journal_d": 55.0, "web_thickness": 18.0, "n_webs": 5}

    and uses ``PARAMS["..."]`` throughout. FLOAT values are dimensions (optimizable); INT values
    are counts/teeth (STRUCTURE -- never touched by the optimizer, which cannot reason about
    topology). Strings/other values are ignored.
  * ``extract_params`` AST-parses the script and returns the dict -- no exec, no LLM.
  * ``override_params`` splices NEW values into the original source (an exact source-segment
    replacement of the dict literal), returning a runnable script for the sandbox. The rest of
    the code is byte-identical, so the optimizer perturbs ONLY the numbers.

Fail-soft by design: a script with no PARAMS dict (or none with float values) simply is not
optimizable -- callers skip, nothing breaks.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass


def param_opt_enabled() -> bool:
    """OPT-IN master flag for the parametric-fit engine (ADR-0073): the default for BOTH halves
    (the author's PARAMS-contract prompt note and the loop's post-loop parameter polish). Off (the
    deployed default) = byte-identical prompts and no polish."""
    return os.environ.get("FORGE_PARAM_OPT") == "1"


def _opt_conditional() -> bool:
    """CONDITIONAL mode (ADR-0108, the council's narrowing): when
    ``FORGE_PARAM_OPT_MODE=conditional``, the engine fires only for a reference-rich part (where the
    proven crankshaft-class gain lives), leaving the reference-less majority byte-identical to
    OFF -- so the ~150 s latency tax is paid only where a gain is plausible. Unset (the default) =
    BLANKET: fire whenever enabled, unchanged. Opt-in, so the live setting is untouched until the
    founder flips it (informed by the drift panel)."""
    return os.environ.get("FORGE_PARAM_OPT_MODE", "").strip().lower() == "conditional"


def param_opt_note_enabled(has_reference: bool = True) -> bool:
    """Gate for STAGE 1 ONLY -- the author's PARAMS-contract prompt note (ADR-0106/0108). Reads
    ``FORGE_PARAM_OPT_NOTE``, falling back to the master ``FORGE_PARAM_OPT``. Split from the polish
    gate because the note is a GLOBAL change to the author's output (it fires at author time, before
    any part is built, and the post-loop validator gate is BLIND to whether it degraded the
    baseline). In CONDITIONAL mode it additionally requires ``has_reference`` (a reference render was
    supplied) -- the cheap pre-author signal that narrows the note to reference-rich parts. BLANKET
    (default) ignores ``has_reference`` -> byte-identical to before."""
    v = os.environ.get("FORGE_PARAM_OPT_NOTE")
    on = v == "1" if v is not None else param_opt_enabled()
    return on and (has_reference or not _opt_conditional())


def param_opt_polish_enabled(has_reference: bool = True) -> bool:
    """Gate for STAGE 3 ONLY -- the post-loop parameter polish (ADR-0106/0108). Reads
    ``FORGE_PARAM_OPT_POLISH``, falling back to the master ``FORGE_PARAM_OPT``. The polish only ever
    REPLACES the baseline when the full validator says its validity tier did not degrade (never
    worse by construction), so it is the safe half to enable. In CONDITIONAL mode it additionally
    requires ``has_reference``; BLANKET (default) ignores it -> byte-identical to before."""
    v = os.environ.get("FORGE_PARAM_OPT_POLISH")
    on = v == "1" if v is not None else param_opt_enabled()
    return on and (has_reference or not _opt_conditional())


@dataclass(frozen=True)
class ParamSpec:
    """The extracted contract of one script: names -> values, and where the dict literal lives."""

    values: dict[str, float | int]  # insertion-ordered, as authored
    start: int  # char offset of the dict literal's "{" in the source
    end: int  # char offset ONE PAST the dict literal's "}"

    @property
    def float_names(self) -> list[str]:
        """The optimizable parameters (floats = dimensions; ints = frozen structure)."""
        return [k for k, v in self.values.items() if isinstance(v, float)]


def _dict_offsets(source: str, node: ast.Dict) -> tuple[int, int]:
    """Char offsets of a Dict literal within ``source`` (line/col -> absolute)."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for ln in lines[:-1]:
        starts.append(starts[-1] + len(ln))
    begin = starts[node.lineno - 1] + node.col_offset
    end_line = node.end_lineno if node.end_lineno is not None else node.lineno
    end = starts[end_line - 1] + (node.end_col_offset or 0)
    return begin, end


def extract_params(code: str) -> ParamSpec | None:
    """The script's ``PARAMS = {...}`` literal as a ParamSpec, or None if absent/not literal.

    Only a MODULE-LEVEL assignment to the exact name ``PARAMS`` whose value is a dict literal of
    string keys to int/float literals counts (negative numbers via unary minus included). Anything
    else -- computed values, non-string keys, a reassigned PARAMS -- returns None (fail-soft: the
    script simply is not optimizable)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "PARAMS"):
            continue
        if not isinstance(node.value, ast.Dict):
            return None
        values: dict[str, float | int] = {}
        for k, v in zip(node.value.keys, node.value.values, strict=True):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                return None
            num = _literal_number(v)
            if num is None:
                continue  # non-numeric entries (strings, bools) are ignored, not fatal
            values[k.value] = num
        if not values:
            return None
        start, end = _dict_offsets(code, node.value)
        return ParamSpec(values=values, start=start, end=end)
    return None


_LIFT_CAP = 24  # bounded: a script with more float literals than this is too ambiguous to lift

# Keyword names whose INT literals are dimensions, not structure (measured 2026-07-21: archived
# strong-red code carries 0 float / 15 int call-arg literals -- local authors write integer
# dims). Counts/teeth/patterns stay frozen: an int is lifted ONLY under these keywords.
_DIM_KEYWORDS = frozenset(
    {
        "radius",
        "height",
        "width",
        "length",
        "depth",
        "thickness",
        "amount",
        "size",
        "diameter",
        "wall",
        "offset",
    }
)


def lift_float_literals(code: str) -> str | None:
    """Rewrite ``code`` so its float literals become a synthetic PARAMS dict (ADR-0117
    follow-on): the delivery mechanism for the ES polish WITHOUT the author-prompt contract
    note (arm PO measured that note regressing authoring -- no_build 1 -> 10, zero strong
    iterations -- while the polish never fired). The lift is deterministic post-processing of
    the FINAL best's code; the polish's acceptance authority (validity-tier non-degrade +
    stated-dim audit) is unchanged, so this can only ever be kept when re-validation improves
    -- the hole-repair-backstop precedent (ADR-0033).

    MEASURED SCOPE LIMIT (TEST_LOG row 63, 2026-07-21): lift+polish did NOT close the
    thin-wall class offline (0/11 archived wall-blocked strong-reds converted; the thinness
    is RELATIONAL between features, not a single knob). Not wired into the loop; kept as
    library machinery (training-data augmentation, offline experiments).

    Returns the rewritten code, or None when: the script already carries a PARAMS dict (the
    authored contract wins), it does not parse, there are no float literals in call
    arguments, or there are more than the cap (too ambiguous)."""
    if extract_params(code) is not None:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    sites: list[tuple[int, int, float]] = []  # (start_offset, end_offset, value)
    lines = code.splitlines(keepends=True)
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln))

    class _V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            kw_args = [(kw.arg, kw.value) for kw in node.keywords]
            pos_args = [(None, a) for a in node.args]
            for kw_name, arg in pos_args + kw_args:
                target = arg
                neg = False
                if (
                    isinstance(target, ast.UnaryOp)
                    and isinstance(target.op, ast.USub)
                    and isinstance(target.operand, ast.Constant)
                ):
                    target, neg = target.operand, True
                if not isinstance(target, ast.Constant):
                    self.generic_visit(arg)
                    continue
                const_val = target.value
                is_dim_int = (
                    isinstance(const_val, int)
                    and not isinstance(const_val, bool)
                    and kw_name is not None
                    and kw_name.lower() in _DIM_KEYWORDS
                )
                if isinstance(const_val, float) or is_dim_int:
                    assert isinstance(const_val, (int, float))
                    src_node: ast.expr = arg if neg else target
                    end_lineno = src_node.end_lineno or src_node.lineno
                    end_col = src_node.end_col_offset or src_node.col_offset
                    start = line_starts[src_node.lineno - 1] + src_node.col_offset
                    end = line_starts[end_lineno - 1] + end_col
                    value = float(-const_val if neg else const_val)
                    sites.append((start, end, value))
                self.generic_visit(arg)
            self.generic_visit(node)

    _V().visit(tree)
    if not sites or len(sites) > _LIFT_CAP:
        return None
    sites.sort()
    out = []
    prev = 0
    names: list[tuple[str, float]] = []
    for i, (start, end, value) in enumerate(sites, 1):
        name = f"p{i}"
        names.append((name, value))
        out.append(code[prev:start])
        out.append(f'PARAMS["{name}"]')
        prev = end
    out.append(code[prev:])
    body = ", ".join(f'"{n}": {v!r}' for n, v in names)
    return f"PARAMS = {{{body}}}\n" + "".join(out)


def _literal_number(node: ast.expr) -> float | int | None:
    """An int/float literal (incl. unary minus), else None. Bools are NOT numbers here."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_number(node.operand)
        return -inner if inner is not None else None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):  # bools are ints to isinstance, but not numbers here
            return None
        return node.value
    return None


def override_params(code: str, spec: ParamSpec, new_values: dict[str, float]) -> str:
    """The SAME script with ``new_values`` spliced into its PARAMS dict literal.

    Unknown names in ``new_values`` are ignored; unmentioned params keep their authored value;
    int-valued params keep their authored value even if named (structure is frozen). Everything
    outside the dict literal is byte-identical."""
    merged: dict[str, float | int] = {}
    for k, v in spec.values.items():
        if isinstance(v, float) and k in new_values:
            merged[k] = float(new_values[k])
        else:
            merged[k] = v
    body = ", ".join(f"{k!r}: {v!r}" for k, v in merged.items())
    return code[: spec.start] + "{" + body + "}" + code[spec.end :]


# Stated-dimension pinning (ADR-0080): the match window and the pinned box half-width.
_MATCH_REL = 0.01
_MATCH_ABS_MM = 0.1
_PIN_SPREAD = 0.02
_PIN_MIN_HALF_MM = 0.1


def _matches_stated(value: float, stated: frozenset[float]) -> bool:
    """True when an authored float traces to a stated number: direct, or the radius idiom
    (build123d APIs take radii -- a stated Ø12 bore is authored 6.0 as often as 12.0)."""
    for s in stated:
        window = max(_MATCH_REL * s, _MATCH_ABS_MM)
        if abs(value - s) <= window or abs(2.0 * value - s) <= window:
            return True
    return False


def pinned_float_names(spec: ParamSpec, stated: frozenset[float] | None) -> list[str]:
    """The float params whose authored values value-match a stated number (ADR-0080).
    Matching is earned from the numbers alone -- the author never declares a pin."""
    if not stated:
        return []
    return [n for n in spec.float_names if _matches_stated(float(spec.values[n]), stated)]


def default_bounds(
    spec: ParamSpec,
    *,
    spread: float = 0.4,
    stated: frozenset[float] | None = None,
    pinned_spread: float = _PIN_SPREAD,
) -> dict[str, tuple[float, float]]:
    """Per-float-param (lo, hi) search bounds: +/-``spread`` around the authored value.

    The authored value came from the brief, so the box keeps the user's stated scale roughly
    sovereign while giving the optimizer real room; dimensions stay strictly positive.

    ADR-0080: a param whose value traces to a STATED number (``stated``, from the user's
    prompt/intake/registry -- never the model's own numbers) is PINNED to +/-``pinned_spread``
    instead: the measured timing-gear incident showed stated dims are otherwise free currency
    the ES trades for wall margin. A pin constrains the FEASIBLE SET -- unlike a loss term it
    cannot be traded against, and erasing the feature never loosens it."""
    pinned = set(pinned_float_names(spec, stated))
    bounds: dict[str, tuple[float, float]] = {}
    for name in spec.float_names:
        v = float(spec.values[name])
        if name in pinned:
            half = max(pinned_spread * abs(v), _PIN_MIN_HALF_MM)
            lo, hi = v - half, v + half
        else:
            lo, hi = v * (1.0 - spread), v * (1.0 + spread)
        if v > 0:
            lo = max(lo, 1e-3)
        bounds[name] = (min(lo, hi), max(lo, hi))
    return bounds


__all__ = [
    "ParamSpec",
    "default_bounds",
    "extract_params",
    "override_params",
    "param_opt_enabled",
    "param_opt_note_enabled",
    "param_opt_polish_enabled",
    "pinned_float_names",
]
