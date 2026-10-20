"""The requirement tree (ADR-0087 S4, founder direction 2026-07-05) — recursive breakdown with
typed, enforced termination.

A problem decomposes ("mount an SSD" -> fit + keep-working -> cables + thermals + ...) until every
leaf lands in a closure the system can actually honor. The MODEL proposes the tree; CODE enforces
the walls (the anchor invariant, applied to reasoning):

  * depth <= 3, <= 24 nodes, fanout <= 6 — an LLM told "keep breaking it down" will happily produce
    a 200-node tree of pseudo-requirements; the caps are the budget (P11) and the quiz guard (P5);
  * every LEAF must carry a closure from the closed vocabulary — ``fact`` (a named fact to gather),
    ``check`` (a validator check id from the playbook's closed set — an invented check demotes),
    ``ask`` (a direct user question), ``design_freedom`` (the engine owns it — the tree STOPS at
    requirements, it does not design), ``not_applicable`` (the over-templating guard), or
    ``unvalidated`` (acknowledged, honest);
  * a domain nomination outside the playbook registry becomes ``other`` and is LOGGED as curation
    demand — handled generically, never improvised.

Every repair/demotion is recorded in ``notes`` — the tree the compiler walks is exactly the tree the
user's card shows, violations included as visible honesty."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.playbooks import CHECK_IDS, PLAYBOOKS
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike

MAX_DEPTH = 3
MAX_NODES = 24
MAX_FANOUT = 6
_DECOMPOSE_MAX_TOKENS = 4000

ClosureKind = Literal["fact", "check", "ask", "design_freedom", "not_applicable", "unvalidated"]


class ProposedNode(BaseModel):
    """One node as the MODEL proposes it (a candidate — validation decides what survives)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    parent_id: str | None = None
    statement: str
    domain: str = "other"
    closure_kind: ClosureKind | None = Field(
        default=None, description="Required on leaves; None on internal nodes."
    )
    closure_detail: str = Field(
        default="",
        description="fact: the fact name; check: the check id; ask: the question; else: why.",
    )


class ProposedTree(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: list[ProposedNode] = Field(default_factory=list)


class Closure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ClosureKind
    detail: str = ""


class RequirementNode(BaseModel):
    """One validated node. Leaves carry a closure; internal nodes carry children."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    statement: str
    domain: str  # a PLAYBOOKS key, or "other"
    closure: Closure | None = None
    children: tuple[RequirementNode, ...] = ()

    def leaves(self) -> list[RequirementNode]:
        if not self.children:
            return [self]
        out: list[RequirementNode] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def touched_domains(self) -> set[str]:
        out = {self.domain} if self.domain != "other" else set()
        for c in self.children:
            out |= c.touched_domains()
        return out


class DecomposeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: RequirementNode | None = None
    notes: list[str] = Field(default_factory=list)  # every repair/demotion, visible


# The canonical fact vocabulary, per domain — generated from the playbooks so the model names
# standard facts EXACTLY as the compiler binds them (naming is coordination: an off-vocabulary
# name cannot resolve from a registry or research template and falls to a generic question).
_FACT_VOCAB = "; ".join(
    f"{name}: {', '.join(t.name for t in pb.facts)}"
    for name, pb in sorted(PLAYBOOKS.items())
    if pb.facts
)

_SYSTEM = f"""You decompose a physical BUILD PROBLEM into a requirement tree for a CAD requirements \
compiler. Break the goal down recursively ("must fit the case" -> what must be true for that) until \
every LEAF terminates in exactly one closure the system can honor:

- fact: ONE named measurable scalar must be gathered (closure_detail = a snake_case fact name). \
Each domain has CANONICAL fact names — when a leaf means one of them, use that EXACT name, one \
leaf per scalar (never bundle dimensions into one fact): {_FACT_VOCAB}. Invent a new snake_case \
name only for a fact this vocabulary does not cover;
- check: a geometric validation the engine runs (closure_detail = one of {list(CHECK_IDS)}; never \
invent a check id);
- ask: only the user can answer (closure_detail = the question, one sentence);
- design_freedom: the CAD engine owns this choice — STOP here, do not design;
- not_applicable: the problem context makes this branch moot (closure_detail = why);
- unvalidated: no instrument exists for this — acknowledge it honestly (closure_detail = why).

Domains you may assign a node: {sorted(PLAYBOOKS)} or "other". Limits: depth <= {MAX_DEPTH} \
(root = depth 0), <= {MAX_NODES} nodes, <= {MAX_FANOUT} children per node. Internal nodes have \
closure_kind = null. One root: the problem goal itself. Decompose what MATTERS for building this \
part; do not pad."""


def _validate(proposed: ProposedTree) -> DecomposeResult:
    """Deterministic walls. Never raises: violations are repaired where safe (demote/prune) and
    every repair is a note — an over-eager tree degrades visibly, never silently."""
    notes: list[str] = []
    seen: dict[str, ProposedNode] = {}
    for n in proposed.nodes:
        if n.node_id in seen:
            notes.append(f"duplicate node id {n.node_id!r} dropped")
            continue
        seen[n.node_id] = n
    roots = [n for n in seen.values() if n.parent_id is None or n.parent_id not in seen]
    true_roots = [n for n in roots if n.parent_id is None]
    if not true_roots:
        return DecomposeResult(root=None, notes=[*notes, "no root node proposed"])
    root_p = true_roots[0]
    for extra in true_roots[1:]:
        notes.append(f"extra root {extra.node_id!r} dropped")
    for orphan in roots:
        if orphan.parent_id is not None:
            notes.append(f"orphan node {orphan.node_id!r} (unknown parent) dropped")

    children_of: dict[str, list[ProposedNode]] = {}
    for n in seen.values():
        if n.parent_id in seen:
            children_of.setdefault(str(n.parent_id), []).append(n)

    count = 0

    def build(p: ProposedNode, depth: int) -> RequirementNode | None:
        nonlocal count
        if count >= MAX_NODES:
            notes.append(f"node cap ({MAX_NODES}) reached; {p.node_id!r} pruned")
            return None
        count += 1
        domain = p.domain if p.domain in PLAYBOOKS else "other"
        if p.domain not in PLAYBOOKS and p.domain != "other":
            notes.append(
                f"unknown domain {p.domain!r} on {p.node_id!r} -> 'other' (curation demand logged)"
            )
        kids_p = children_of.get(p.node_id, [])
        if len(kids_p) > MAX_FANOUT:
            notes.append(f"fanout cap on {p.node_id!r}: {len(kids_p) - MAX_FANOUT} children pruned")
            kids_p = kids_p[:MAX_FANOUT]
        if depth >= MAX_DEPTH and kids_p:
            notes.append(f"depth cap under {p.node_id!r}: subtree demoted to an unvalidated leaf")
            kids_p = []
            return RequirementNode(
                node_id=p.node_id,
                statement=p.statement,
                domain=domain,
                closure=Closure(kind="unvalidated", detail="depth cap: not decomposed further"),
            )
        kids = [k for k in (build(c, depth + 1) for c in kids_p) if k is not None]
        if kids:
            return RequirementNode(
                node_id=p.node_id, statement=p.statement, domain=domain, children=tuple(kids)
            )
        # a leaf MUST close; repair dishonest/invalid closures downward, never upward
        kind: ClosureKind = p.closure_kind or "unvalidated"
        detail = p.closure_detail
        if p.closure_kind is None:
            notes.append(f"leaf {p.node_id!r} had no closure -> unvalidated")
            detail = detail or "no closure proposed"
        if kind == "check":
            allowed = (
                {c.check_id for c in PLAYBOOKS[domain].checks} if domain in PLAYBOOKS else set()
            )
            if detail not in allowed:
                notes.append(
                    f"leaf {p.node_id!r}: check {detail!r} is not in the {domain} playbook -> unvalidated"
                )
                kind, detail = "unvalidated", f"proposed check {detail!r} unavailable for {domain}"
        return RequirementNode(
            node_id=p.node_id,
            statement=p.statement,
            domain=domain,
            closure=Closure(kind=kind, detail=detail),
        )

    root = build(root_p, 0)
    return DecomposeResult(root=root, notes=notes)


def decompose_problem(
    problem: str,
    *,
    client: LLMClientLike,
    model: str,
    context: str = "",
    max_tokens: int = _DECOMPOSE_MAX_TOKENS,
) -> tuple[DecomposeResult, LLMUsage]:
    """One brain pass proposes the tree; the deterministic walls validate it."""
    ctx = f"\nKnown context:\n{context.strip()}\n" if context.strip() else ""
    user = f"Problem:\n{problem.strip()}\n{ctx}\nProduce the requirement tree."
    proposed, usage = client.generate_structured(
        system=_SYSTEM,
        user=user,
        output_model=ProposedTree,
        model=model,
        max_tokens=max_tokens,
    )
    return _validate(proposed), usage


__all__ = [
    "MAX_DEPTH",
    "MAX_FANOUT",
    "MAX_NODES",
    "Closure",
    "ClosureKind",
    "DecomposeResult",
    "ProposedNode",
    "ProposedTree",
    "RequirementNode",
    "decompose_problem",
]
