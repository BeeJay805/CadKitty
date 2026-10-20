"""Fail-closed research (ADR-0087 S3): the deterministic verifier DELETES any fact it cannot pin to
a fetched quote (P1), product variants surface as ambiguity (P2), fetch dates ride (P3), units
normalize (P4), and researched conflicts are visible (P10) — no branch trusts the model's word."""

from __future__ import annotations

from typing import Any

from forge_mind.invention.requirements import FactLedger
from forge_mind.invention.research import (
    Document,
    FakeResearchProvider,
    ProposedFact,
    ResearchProposal,
    detect_variants,
    research_facts,
    verify_proposed_fact,
)
from forge_mind.llm.cost import LLMUsage

_DOC = Document(
    url="https://example.com/nr200",
    title="Cooler Master NR200 spec",
    text="The NR200 supports 2.5 inch drives. The drive bay depth is 4 inches. Fan clearance is 80 mm.",
    fetched_at="2026-07-05",
)


class _StubClient:
    """Returns a canned ResearchProposal — the model's proposal, which the VERIFIER still judges."""

    def __init__(self, proposal: ResearchProposal) -> None:
        self.proposal = proposal

    def generate_structured(self, *, model: str, **_: Any) -> tuple[ResearchProposal, LLMUsage]:
        return self.proposal, LLMUsage(model=model, input_tokens=1, output_tokens=1)


# --- the verifier: the no-false-PASS heart (P1) ---------------------------------------------------


def test_verifier_admits_a_quoted_fact() -> None:
    p = ProposedFact(
        name="bay_depth",
        value=4.0,
        unit="inches",
        quote="drive bay depth is 4 inches",
        url=_DOC.url,
    )
    fact = verify_proposed_fact(p, [_DOC])
    assert fact is not None
    assert fact.source == "researched" and fact.citation == _DOC.url
    assert fact.as_mm_or_g == (101.6, "mm")  # P4: 4 inches normalizes deterministically


def test_verifier_drops_a_hallucinated_quote() -> None:
    # a quote that is NOT in the document = a fabrication -> deleted (P1)
    p = ProposedFact(
        name="bay_depth",
        value=5.0,
        unit="inches",
        quote="drive bay depth is 5 inches",
        url=_DOC.url,
    )
    assert verify_proposed_fact(p, [_DOC]) is None


def test_verifier_drops_when_value_not_in_quote() -> None:
    # the quote is real but the model's value is not IN it (number laundering) -> deleted
    p = ProposedFact(
        name="fan_clearance", value=120.0, unit="mm", quote="Fan clearance is 80 mm.", url=_DOC.url
    )
    assert verify_proposed_fact(p, [_DOC]) is None


def test_verifier_drops_when_url_not_fetched() -> None:
    p = ProposedFact(
        name="bay_depth",
        value=4.0,
        unit="inches",
        quote="drive bay depth is 4 inches",
        url="https://evil.example/made-up",
    )
    assert verify_proposed_fact(p, [_DOC]) is None


def test_verifier_drops_when_unit_absent_from_quote() -> None:
    # value present, but the unit the model claims is not in the span -> deleted
    p = ProposedFact(name="x", value=4.0, unit="cm", quote="supports 2.5 inch drives", url=_DOC.url)
    assert verify_proposed_fact(p, [_DOC]) is None


# --- orchestration: research_facts keeps only what verifies -------------------------------------


def test_research_facts_returns_only_verified_and_lists_dropped() -> None:
    proposal = ResearchProposal(
        facts=[
            ProposedFact(
                name="bay_depth",
                value=4.0,
                unit="inches",
                quote="drive bay depth is 4 inches",
                url=_DOC.url,
            ),
            ProposedFact(
                name="fantasy",
                value=999.0,
                unit="mm",
                quote="the case has 999 mm of magic",
                url=_DOC.url,
            ),
        ]
    )
    provider = FakeResearchProvider({"nr200": [_DOC]})
    result, _usage = research_facts(
        "NR200 drive bay depth",
        provider=provider,
        client=_StubClient(proposal),
        model="m",
        entity="NR200",
    )
    assert [f.name for f in result.facts] == ["bay_depth"]  # the fantasy fact is gone
    assert any("fantasy" in d for d in result.dropped)


def test_research_facts_no_documents_is_honest() -> None:
    provider = FakeResearchProvider({"nr200": [_DOC]})
    result, _ = research_facts(
        "totally unrelated widget",
        provider=provider,
        client=_StubClient(ResearchProposal()),
        model="m",
    )
    assert result.facts == [] and any("no documents" in n for n in result.notes)


# --- variant ambiguity (P2) ----------------------------------------------------------------------


def test_variant_ambiguity_surfaced_never_picked() -> None:
    docs = [
        Document(
            url="a", title="Cooler Master NR200 review", text="NR200 ...", fetched_at="2026-07-05"
        ),
        Document(
            url="b", title="Cooler Master NR200P specs", text="NR200P ...", fetched_at="2026-07-05"
        ),
    ]
    variants = detect_variants("NR200", docs)
    assert set(variants) >= {"NR200", "NR200P"}
    provider = FakeResearchProvider({"nr200": docs})
    result, _ = research_facts(
        "NR200 dimensions",
        provider=provider,
        client=_StubClient(ResearchProposal()),
        model="m",
        entity="NR200",
    )
    assert len(result.variant_ambiguity) > 1
    assert any("variants" in n for n in result.notes)


# --- P3 fetch date, P10 conflict --------------------------------------------------------------


def test_fetched_at_rides_the_document() -> None:
    assert _DOC.fetched_at == "2026-07-05"  # surfaced so a stale citation is visible


def test_conflicting_researched_facts_are_visible() -> None:
    # two documents disagree -> both verify, and the ledger surfaces the conflict (never averaged)
    doc2 = Document(
        url="https://example.com/other",
        title="other",
        text="The drive bay depth is 100 mm.",
        fetched_at="2026-07-05",
    )
    p1 = ProposedFact(
        name="bay_depth",
        value=4.0,
        unit="inches",
        quote="drive bay depth is 4 inches",
        url=_DOC.url,
    )
    p2 = ProposedFact(
        name="bay_depth", value=100.0, unit="mm", quote="drive bay depth is 100 mm", url=doc2.url
    )
    f1 = verify_proposed_fact(p1, [_DOC])
    f2 = verify_proposed_fact(p2, [doc2])
    assert f1 is not None and f2 is not None
    ledger = FactLedger(facts=(f1, f2))
    assert len(ledger.conflicts()) == 1  # 101.6 mm vs 100.0 mm for bay_depth


# --- S6b: the live provider (offline via transport injection) + the factory ---------------------


def test_tavily_provider_parses_results_offline() -> None:
    import httpx

    from forge_mind.invention.research import TavilyResearchProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.tavily.com/search"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "raw_content": "The bay depth is 4 inches.",
                    },
                    {"url": "https://example.com/b", "title": "B", "content": "snippet only"},
                    {"url": "", "title": "dropped", "raw_content": "no url -> skipped"},
                    {"url": "https://example.com/c", "title": "dropped", "raw_content": ""},
                ]
            },
        )

    provider = TavilyResearchProvider("key", transport=httpx.MockTransport(handler))
    docs = provider.search("NR200 bay depth")
    assert [d.url for d in docs] == ["https://example.com/a", "https://example.com/b"]
    assert docs[0].text == "The bay depth is 4 inches."
    assert docs[1].text == "snippet only"  # content fallback when raw_content is absent
    assert docs[0].fetched_at  # the fetch date rides every citation (P3)


def test_tavily_provider_fails_closed_on_api_error() -> None:
    import httpx

    from forge_mind.invention.research import TavilyResearchProvider

    provider = TavilyResearchProvider(
        "key", transport=httpx.MockTransport(lambda _req: httpx.Response(500))
    )
    assert provider.search("anything") == []  # no documents -> facts route to ASK, never faked


def test_get_research_provider_resolution(monkeypatch) -> None:
    from forge_mind.invention.research import TavilyResearchProvider, get_research_provider

    monkeypatch.delenv("FORGE_RESEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert get_research_provider() is None  # unset -> BLOCKED, never faked
    monkeypatch.setenv("FORGE_RESEARCH_PROVIDER", "tavily")
    assert get_research_provider() is None  # provider named but no key -> still blocked
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert isinstance(get_research_provider(), TavilyResearchProvider)
    monkeypatch.setenv("FORGE_RESEARCH_PROVIDER", "unknown-thing")
    assert get_research_provider() is None  # an unknown provider never half-works
