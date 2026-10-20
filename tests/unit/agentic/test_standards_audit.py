"""Standards-audit (ADR-0009 enforcement): the author must FETCH every brief-declared standard
dimension from the registry. A declared-but-hand-modeled standard is REFUSED; what it fetched is
CITED. Pure AST + registry -- no geometry, no LLM."""

from __future__ import annotations

from forge_mind.agentic.standards_audit import audit_standards, cite_standard, fetched_standards


def test_fetches_clearance_hole() -> None:
    uses = fetched_standards("d = clearance_hole_d('M3')")
    assert len(uses) == 1
    assert uses[0].category == "clearance_hole"
    assert uses[0].designation == "M3"


def test_fetches_through_module_attr_and_multiple() -> None:
    code = "a = helpers.clearance_hole_d('M5')\nb = bearing_seat_dims('608')"
    cats = {(u.category, u.designation) for u in fetched_standards(code)}
    assert ("clearance_hole", "M5") in cats
    assert ("bearing_seat", "608") in cats


def test_nested_call_is_found() -> None:
    # clearance_hole_d is nested inside a geometry-helper call -- ast.walk must still find it.
    code = "result = plate - bolt_holes(pcd=40, hole_d=clearance_hole_d('M3'), count=4)"
    assert any(u.designation == "M3" for u in fetched_standards(code))


def test_syntax_error_is_safe() -> None:
    assert fetched_standards("def (((") == []


def test_declared_and_fetched_passes_with_cite() -> None:
    code = "result = plate - bolt_holes(pcd=40, hole_d=clearance_hole_d('M3'), count=4)"
    cites, fails = audit_standards(code, ["M3"])
    assert fails == []
    assert any("M3" in c and "3.4" in c for c in cites)


def test_declared_but_handmodeled_is_refused() -> None:
    # The author hand-modeled a 3.0mm hole and never fetched M3 -> the part is REFUSED.
    code = "result = plate - bolt_holes(pcd=40, hole_d=3.0, count=4)"
    cites, fails = audit_standards(code, ["M3"])
    assert fails, "a brief-declared M3 the code never fetched must be refused"
    assert "M3" in fails[0]
    assert cites == []


def test_match_is_case_insensitive() -> None:
    _, fails = audit_standards("x = clearance_hole_d('m3')", ["M3"])
    assert fails == []


def test_no_declared_standards_means_no_gate() -> None:
    code = "result = plate - bolt_holes(pcd=40, hole_d=3.0, count=4)"
    cites, fails = audit_standards(code, [])
    assert fails == []
    assert cites == []


def test_cite_standard_resolves_value_and_source() -> None:
    cite = cite_standard("clearance_hole", "M3")
    assert cite is not None
    assert "M3" in cite and "3.4" in cite


def test_declared_m10_now_supplied_is_still_enforced() -> None:
    # M10-M16 were added to the registry (ADR-0067): a declared-but-hand-modeled M10 is now
    # SUPPLIABLE, so the audit still refuses it and points at the (now-working) helper.
    code = "result = plate - bolt_holes(pcd=88, hole_d=10.5, count=8)"
    _, fails = audit_standards(code, ["M10"])
    assert fails and "M10" in fails[0]


def test_declared_unsupplied_size_is_not_a_crash_trap() -> None:
    # An UNSUPPLIED size (no registry entry) must NOT be refused: ordering the author to call the
    # helper would raise and crash the build (the M10 poison-pill, generalized). Hand-model allowed.
    code = "result = plate - bolt_holes(pcd=88, hole_d=21.0, count=8)"  # M20, not in any registry
    _, fails = audit_standards(code, ["M20"])
    assert fails == []


def test_supplied_and_unsupplied_declared_together() -> None:
    # M10 supplied -> refused (fetch it); M20 unsupplied -> skipped. Exactly one fail, for M10.
    _, fails = audit_standards("result = 1", ["M10", "M20"])
    assert len(fails) == 1 and "M10" in fails[0]
