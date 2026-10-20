"""Domain playbooks (ADR-0087 S4): the seed five exist, ceilings are verbatim where honesty
requires them, checks use only the closed vocabulary, and templates are format-safe."""

from __future__ import annotations

from forge_mind.invention.playbooks import CHECK_IDS, PLAYBOOKS, classify_domain


def test_seed_domains_present() -> None:
    assert set(PLAYBOOKS) == {
        "fit_envelope",
        "mount_interface",
        "thermal_airflow",
        "cable_routing",
        "access_service",
    }


def test_ceilings_where_no_instrument_exists() -> None:
    assert "NOT simulated" in (PLAYBOOKS["thermal_airflow"].honesty_ceiling or "")
    assert PLAYBOOKS["cable_routing"].honesty_ceiling is not None
    assert PLAYBOOKS["access_service"].honesty_ceiling is not None


def test_checks_use_only_the_closed_vocabulary() -> None:
    for pb in PLAYBOOKS.values():
        for check in pb.checks:
            assert check.check_id in CHECK_IDS
            assert check.params_from, f"{pb.domain}/{check.check_id} has no parameterizing facts"


def test_templates_are_format_safe() -> None:
    # only the {entity} placeholder is allowed; stray braces would crash the ask rendering
    for pb in PLAYBOOKS.values():
        for t in pb.facts:
            assert "{" not in t.ask.replace("{entity}", ""), f"{pb.domain}/{t.name} ask"
            if t.research_query:
                assert "{" not in t.research_query.replace("{entity}", "")


def test_classifier_assist() -> None:
    assert classify_domain("the cooling fan airflow must stay open") == "thermal_airflow"
    assert classify_domain("route the SATA cable to the connector") == "cable_routing"
    assert classify_domain("a poem about geese") is None
