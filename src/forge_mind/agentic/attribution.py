"""Line-level failure attribution (research idea 6, from the bench row-18 plateau class).

The measured failure: a part BUILDS but the loop stalls (plateau) because the author reads the
validator's prose findings ("MISSING 2 x Ø4.5 ('set-screw holes')") and cannot connect them back
to its OWN code -- so it edits the wrong block, or nothing, and the same finding repeats for
iterations. This module maps each actionable finding to the PARAMS keys and code statements that
build that feature, and hands the author a CODE MAP so the revision is a targeted edit.

Design law: PRECISION over recall. A wrong pointer is worse than none (it sends the author to
edit code that was fine), so every mapping is high-confidence or omitted, and the whole thing
fail-softs to None. The one strong claim it makes is the SAFE one: when a brief feature is
reported MISSING and NO line references it, "your code builds no such feature -- add it."
Advisory context only; never a gate.
"""

from __future__ import annotations

import os
import re
from typing import Any

from forge_mind.agentic.params import extract_params


def line_attribution_enabled() -> bool:
    """Default ON (precision-first, so it can only add correct pointers or stay silent -- a wrong
    pointer never ships); FORGE_LINE_ATTRIBUTION=0 disables it for an A/B or a regression."""
    return os.environ.get("FORGE_LINE_ATTRIBUTION", "1") != "0"


# Generic words that carry no attribution signal -- a match on these would point everywhere.
_STOP = frozenset(
    {
        "hole",
        "holes",
        "bore",
        "bores",
        "the",
        "a",
        "an",
        "of",
        "for",
        "and",
        "mm",
        "x",
        "through",
        "part",
        "feature",
        "features",
        "with",
        "on",
        "in",
        "at",
        "one",
        "two",
        "cut",
        "face",
        "side",
    }
)

# Findings whose text quotes a feature LABEL the author can act on. Wall/envelope/watertight are
# whole-part properties, not one code block, so they are deliberately not attributed.
_ACTIONABLE = ("MISSING FEATURES", "MEASURED BORES", "SIZE MISMATCH", "RADIAL FEATURES")

_LABEL_RE = re.compile(r"'([^']{2,40})'")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens(text: str) -> set[str]:
    """Distinctive content words of a label: lowercased, split on non-alnum and snake/camel
    boundaries, generic + short words dropped."""
    parts = re.split(r"[^A-Za-z0-9]+", text)
    words: set[str] = set()
    for p in parts:
        # split camelCase / digit runs into their word pieces
        for w in re.findall(r"[A-Za-z]+", p):
            wl = w.lower()
            if len(wl) >= 3 and wl not in _STOP:
                words.add(wl)
    return words


def _line_words(line: str) -> set[str]:
    """Identifier + comment words on a code line, snake_case split, lowercased."""
    words: set[str] = set()
    for ident in _IDENT_RE.findall(line):
        for w in ident.split("_"):
            if len(w) >= 3:
                words.add(w.lower())
    return words


def _labels_from_issue(issue: str) -> list[str]:
    return _LABEL_RE.findall(issue)


def attribute_issues(issues: list[str], code: str, brief: Any = None) -> str | None:
    """Map actionable validator findings to the author's code. Returns a CODE MAP block, or None
    when nothing maps confidently (fail-soft on any error)."""
    try:
        actionable = [i for i in issues if any(i.startswith(p) for p in _ACTIONABLE)]
        if not actionable:
            return None
        code_lines = code.splitlines()
        spec = extract_params(code)
        param_names = list(spec.values.keys()) if spec is not None else []

        # Precompute per-line word sets once.
        line_words = [(_i + 1, _line_words(ln), ln.strip()) for _i, ln in enumerate(code_lines)]
        param_words = {name: _line_words(name) for name in param_names}

        mapped: list[str] = []
        seen_labels: set[str] = set()
        for issue in actionable:
            for label in _labels_from_issue(issue):
                key = label.lower()
                if key in seen_labels:
                    continue
                toks = _tokens(label)
                if not toks:
                    continue
                seen_labels.add(key)

                hit_params = [n for n in param_names if toks & param_words[n]]
                hit_lines = [
                    (ln_no, snippet)
                    for (ln_no, words, snippet) in line_words
                    if toks & words and snippet
                ]
                # PRECISION guard: a token that lights up many lines is generic FOR THIS script
                # (e.g. "plate" in a plate part) -- not a locator. Drop those matches.
                if len(hit_lines) > 4:
                    hit_lines = []

                if hit_params or hit_lines:
                    where = []
                    if hit_params:
                        where.append("PARAMS " + ", ".join(f"`{n}`" for n in hit_params[:4]))
                    if hit_lines:
                        where.append("; ".join(f"line {n} `{s[:60]}`" for n, s in hit_lines[:3]))
                    mapped.append(f"  - '{label}' is built by: {' | '.join(where)}")
                elif issue.startswith("MISSING FEATURES") or "MISSING" in issue:
                    # The SAFE strong claim: the finding says this feature is short, and NO line
                    # references it -- so the code does not build it at all.
                    mapped.append(
                        f"  - '{label}' has NO block in your code -- it is not built at all; ADD it."
                    )

        if not mapped:
            return None
        return (
            "CODE MAP (which of YOUR lines each finding points to -- edit exactly these, "
            "keep everything else):\n" + "\n".join(mapped)
        )
    except Exception:
        return None


__all__ = ["attribute_issues", "line_attribution_enabled"]
