"""The requirement tree's deterministic walls (ADR-0087 S4): the model proposes, code enforces —
caps, typed leaves, closed check vocabulary, unknown domains demoted and logged. Every repair is a
visible note, never a silent fix."""

from __future__ import annotations

from forge_mind.invention.decompose import (
    MAX_NODES,
    ProposedNode,
    ProposedTree,
    _validate,
)


def _n(nid: str, parent: str | None, stmt: str, **kw: object) -> ProposedNode:
    return ProposedNode(node_id=nid, parent_id=parent, statement=stmt, **kw)  # type: ignore[arg-type]


def test_valid_tree_builds() -> None:
    tree = ProposedTree(
        nodes=[
            _n("r", None, "mount an SSD"),
            _n(
                "a",
                "r",
                "must fit",
                domain="fit_envelope",
                closure_kind="fact",
                closure_detail="available_height",
            ),
            _n(
                "b",
                "r",
                "airflow stays open",
                domain="thermal_airflow",
                closure_kind="check",
                closure_detail="keepout_clear",
            ),
        ]
    )
    result = _validate(tree)
    assert result.root is not None and len(result.root.children) == 2
    assert result.notes == []


def test_leaf_without_closure_demotes_to_unvalidated() -> None:
    result = _validate(ProposedTree(nodes=[_n("r", None, "goal"), _n("a", "r", "vague wish")]))
    assert result.root is not None
    (leaf,) = result.root.children
    assert leaf.closure is not None and leaf.closure.kind == "unvalidated"
    assert any("no closure" in n for n in result.notes)


def test_invented_check_id_demotes() -> None:
    tree = ProposedTree(
        nodes=[
            _n("r", None, "goal"),
            _n(
                "a",
                "r",
                "thermals right",
                domain="thermal_airflow",
                closure_kind="check",
                closure_detail="cfd_simulation",
            ),
        ]
    )
    result = _validate(tree)
    assert result.root is not None
    (leaf,) = result.root.children
    assert leaf.closure is not None and leaf.closure.kind == "unvalidated"
    assert any("cfd_simulation" in n for n in result.notes)


def test_unknown_domain_becomes_other_and_logs_demand() -> None:
    tree = ProposedTree(
        nodes=[
            _n("r", None, "goal"),
            _n(
                "a",
                "r",
                "must not rattle",
                domain="vibration",
                closure_kind="unvalidated",
                closure_detail="no instrument",
            ),
        ]
    )
    result = _validate(tree)
    assert result.root is not None
    assert result.root.children[0].domain == "other"
    assert any("vibration" in n and "curation demand" in n for n in result.notes)


def test_depth_cap_demotes_the_subtree() -> None:
    tree = ProposedTree(
        nodes=[
            _n("r", None, "d0"),
            _n("a", "r", "d1"),
            _n("b", "a", "d2"),
            _n("c", "b", "d3 has kids"),
            _n("d", "c", "d4 too deep", closure_kind="fact", closure_detail="x"),
        ]
    )
    result = _validate(tree)
    assert result.root is not None
    node = result.root.children[0].children[0].children[0]
    assert node.closure is not None and node.closure.kind == "unvalidated"
    assert any("depth cap" in n for n in result.notes)


def test_duplicate_orphan_extra_root_and_caps() -> None:
    nodes = [
        _n("r", None, "root"),
        _n("r2", None, "another root", closure_kind="unvalidated", closure_detail="x"),
        _n("a", "r", "dup", closure_kind="fact", closure_detail="f"),
        _n("a", "r", "dup again", closure_kind="fact", closure_detail="f"),
        _n("o", "ghost", "orphan", closure_kind="fact", closure_detail="f"),
    ]
    nodes += [
        _n(f"k{i}", "r", f"child {i}", closure_kind="fact", closure_detail=f"f{i}")
        for i in range(7)
    ]
    result = _validate(ProposedTree(nodes=nodes))
    assert result.root is not None
    joined = "\n".join(result.notes)
    assert "duplicate node id" in joined
    assert "extra root" in joined
    assert "orphan" in joined
    assert "fanout cap" in joined
    assert len(result.root.children) <= 6


def test_node_cap_prunes_with_note() -> None:
    nodes = [_n("r", None, "root")]
    for i in range(MAX_NODES + 4):
        nodes.append(_n(f"g{i}", "r", f"leaf {i}", closure_kind="fact", closure_detail=f"f{i}"))
    # fanout cap trims to 6 first, so exceed via a chain of wide layers instead
    nodes = [_n("r", None, "root")]
    parents = ["r"]
    i = 0
    while len(nodes) < MAX_NODES + 6:
        new_parents = []
        for p in parents:
            for _j in range(3):
                nid = f"n{i}"
                i += 1
                nodes.append(_n(nid, p, f"node {nid}", closure_kind="fact", closure_detail="f"))
                new_parents.append(nid)
        parents = new_parents
    result = _validate(ProposedTree(nodes=nodes))
    assert any("node cap" in n for n in result.notes)


def test_no_root_is_honest() -> None:
    result = _validate(ProposedTree(nodes=[_n("a", "missing", "orphan only")]))
    assert result.root is None
    assert any("no root" in n for n in result.notes)
