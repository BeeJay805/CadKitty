"""Fail-closed cited research (ADR-0087 S3) — the compiler learns about the world without lying.

A problem like "mount an SSD in my NR200" needs facts about a NAMED product the registry does not
carry. The model must NOT supply them from memory (ADR-0087 P1: that is exactly the hallucinated-spec
failure). Instead:

  1. a research PROVIDER fetches documents for a query (the ``ResearchProviderLike`` seam — a real
     web-search API in prod, a canned corpus in tests; both return the SAME ``Document`` shape);
  2. the brain PROPOSES facts, each pinned to an exact ``quote`` and the ``url`` it came from — it
     reads the fetched text, it does not recall;
  3. a DETERMINISTIC verifier admits a fact ONLY when the quote literally appears in that fetched
     document AND the value + unit literally appear in the quote AND the url was actually fetched.
     Anything else is DROPPED (never a silent guess) — this is the vendor-gate discipline (L2), and
     it is code, not a prompt, so a persuasive hallucination cannot survive it.

Admitted facts are :class:`~forge_mind.invention.requirements.Fact` at ``source="researched"`` ->
the SOURCED tier that can never launder into STATED, carrying the URL citation and the fetched quote
(S1). Multiple product VARIANTS in the results are surfaced as an ambiguity (P2), never silently
picked; every result carries its ``fetched_at`` date (P3). No provider configured -> research is
BLOCKED (the toolbelt preflight says so), never faked.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.requirements import Fact
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_VALUE_EPS = 1e-4
_RESEARCH_MAX_TOKENS = 3000


class Document(BaseModel):
    """One fetched result. ``text`` is the plain text a fact's quote must be found IN; ``fetched_at``
    is the retrieval date (ISO), surfaced so a stale citation is visible (P3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    title: str
    text: str
    fetched_at: str = ""


class ResearchProviderLike(Protocol):
    """The structural contract a research backend satisfies. A prod provider hits a web-search API;
    the test provider serves a canned corpus. Both return the SAME ``Document`` list."""

    def search(self, query: str) -> list[Document]: ...


class FakeResearchProvider:
    """A canned corpus keyed by substring match, for offline tests and the compiler's golden
    scenarios (no spend, deterministic). ``by_query`` maps a lowercase substring -> its documents;
    a search returns the union of every entry whose key is in the (lowercased) query."""

    def __init__(self, by_query: dict[str, list[Document]]) -> None:
        self._by_query = {k.lower(): v for k, v in by_query.items()}

    def search(self, query: str) -> list[Document]:
        q = query.lower()
        out: list[Document] = []
        seen: set[str] = set()
        for key, docs in self._by_query.items():
            if key in q:
                for d in docs:
                    if d.url not in seen:
                        seen.add(d.url)
                        out.append(d)
        return out


class ProposedFact(BaseModel):
    """A fact the brain proposes FROM a fetched document. It is a CANDIDATE only — the verifier, not
    the model, decides whether it becomes a Fact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Short snake_case requirement name, e.g. 'bay_depth'.")
    value: float
    unit: str
    quote: str = Field(..., description="The EXACT span from the document that states this value.")
    url: str = Field(..., description="The url of the document the quote came from.")


class ResearchProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: list[ProposedFact] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Verified facts + everything the caller must SEE: what was dropped, and variant ambiguity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: list[Fact] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)  # 'name: reason' for each rejected proposal
    variant_ambiguity: list[str] = Field(
        default_factory=list
    )  # distinct product variants seen (P2)
    notes: list[str] = Field(default_factory=list)


_SYSTEM = """You extract mechanical FACTS from the provided documents for a CAD requirements \
compiler. Read ONLY the documents given; never use prior knowledge. For each fact you can support:
- name: a short snake_case requirement name (e.g. bay_depth, fan_clearance);
- value + unit: the number and its unit, EXACTLY as the document states them;
- quote: the exact span of text from the document that contains that value (verbatim — it will be \
checked character-for-character against the document);
- url: the url of the document the quote is from.
If the documents do not state a value, do NOT propose it. A fact you cannot quote is worthless: a \
downstream verifier will DELETE any fact whose quote is not found verbatim in its document, or whose \
value/unit is not in the quote. Propose only what you can quote. Propose a fact ONLY when the \
document's statement is genuinely about that fact's subject — never map a number about one thing \
(a fan's clearance) onto a differently-subjected fact (a drive bay's clearance)."""


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _value_in_text(value: float, text: str) -> bool:
    return any(abs(float(n) - value) <= _VALUE_EPS for n in _NUM_RE.findall(text))


def verify_proposed_fact(proposed: ProposedFact, documents: list[Document]) -> Fact | None:
    """DETERMINISTIC admission (the no-false-PASS heart). A proposal becomes a Fact ONLY if:
    (1) its url was actually fetched; (2) its quote appears verbatim (whitespace/case-normalized) in
    that document; (3) the value appears as a number in the quote; (4) the unit token appears in the
    quote. Otherwise -> None (dropped). No branch trusts the model's word."""
    by_url = {d.url: d for d in documents}
    doc = by_url.get(proposed.url)
    if doc is None:
        return None
    nq = _norm(proposed.quote)
    if not nq or nq not in _norm(doc.text):
        return None
    if not _value_in_text(proposed.value, proposed.quote):
        return None
    if proposed.unit.strip().lower() not in nq:
        return None
    return Fact(
        name=proposed.name,
        value=proposed.value,
        unit=proposed.unit,
        source="researched",
        citation=proposed.url,
        quote=proposed.quote.strip(),
    )


_VARIANT_RE = re.compile(r"\b([A-Za-z]{1,6}\d{2,4}[A-Za-z0-9+\-]*)\b")


def detect_variants(entity: str, documents: list[Document]) -> list[str]:
    """Distinct product-model tokens sharing the entity's alphanumeric stem, across the doc titles
    (P2). 'NR200' with docs titled 'NR200', 'NR200P', 'NR200 MAX' -> ['NR200', 'NR200P'] etc. -> the
    caller asks WHICH, never silently picks one."""
    stem = re.sub(r"[^A-Za-z0-9]", "", entity).lower()[:5]
    if len(stem) < 3:
        return []
    seen: dict[str, None] = {}
    for d in documents:
        for tok in _VARIANT_RE.findall(d.title):
            if re.sub(r"[^A-Za-z0-9]", "", tok).lower().startswith(stem):
                seen.setdefault(tok, None)
    return list(seen)


def propose_research_facts(
    question: str,
    documents: list[Document],
    *,
    client: LLMClientLike,
    model: str,
    wanted_names: Sequence[str] = (),
    max_tokens: int = _RESEARCH_MAX_TOKENS,
) -> tuple[ResearchProposal, LLMUsage]:
    """One brain pass: propose quotable facts from the documents. Pure of side effects beyond the
    call; the verifier — not this — decides admission. ``wanted_names`` tells the brain the exact
    fact names the compiler binds on (S6b/S7: without them the brain names facts freely and the
    compiler's template match silently misses — naming is coordination, not provenance; the
    deterministic quote/value verifier is unchanged by it)."""
    corpus = "\n\n".join(f"URL: {d.url}\nTITLE: {d.title}\nTEXT: {d.text}" for d in documents)
    naming = (
        "\nName each fact EXACTLY one of these snake_case names (propose only the ones the "
        f"documents support): {', '.join(wanted_names)}."
        if wanted_names
        else ""
    )
    user = (
        f"Question: {question}\n\nDocuments:\n{corpus}\n{naming}\nPropose only facts you can quote."
    )
    return client.generate_structured(
        system=_SYSTEM,
        user=user,
        output_model=ResearchProposal,
        model=model,
        max_tokens=max_tokens,
    )


def research_facts(
    question: str,
    *,
    provider: ResearchProviderLike,
    client: LLMClientLike,
    model: str,
    entity: str = "",
    wanted_names: Sequence[str] = (),
) -> tuple[ResearchResult, LLMUsage]:
    """Search -> propose -> VERIFY. Returns only verified facts, with the dropped candidates and any
    product-variant ambiguity surfaced. ``entity`` (the named product) enables variant detection;
    ``wanted_names`` binds proposed fact names to the compiler's templates (naming only — admission
    stays with the deterministic verifier)."""
    documents = provider.search(question)
    if not documents:
        zero = LLMUsage(model="", input_tokens=0, output_tokens=0)
        return ResearchResult(notes=[f"no documents found for: {question}"]), zero
    proposal, usage = propose_research_facts(
        question, documents, client=client, model=model, wanted_names=wanted_names
    )
    verified: list[Fact] = []
    dropped: list[str] = []
    for p in proposal.facts:
        fact = verify_proposed_fact(p, documents)
        if fact is not None:
            verified.append(fact)
        else:
            dropped.append(f"{p.name}: not verifiable against the fetched documents (dropped)")
    variants = detect_variants(entity, documents) if entity else []
    ambiguity = variants if len(variants) > 1 else []
    result = ResearchResult(
        facts=verified,
        dropped=dropped,
        variant_ambiguity=ambiguity,
        notes=(
            [f"multiple product variants found ({', '.join(ambiguity)}); confirm which one"]
            if ambiguity
            else []
        ),
    )
    return result, usage


class TavilyResearchProvider:
    """The live web-search provider (S6b): one POST to the Tavily search API per query, returning
    fetched page content (``raw_content``) — the L2 quote-verifier needs REAL fetched text to check
    quotes against; a snippets-only API cannot anchor a quote. Fail-closed: any transport/API error
    returns NO documents (facts route to ASK — the form asks the user; nothing is ever faked)."""

    _URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        *,
        max_results: int = 5,
        timeout_s: float = 20.0,
        transport: Any | None = None,  # httpx transport injection for offline tests
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._timeout_s = timeout_s
        self._transport = transport

    def search(self, query: str) -> list[Document]:
        import httpx  # lazy: httpx is a [server]-extra dependency, not a core one

        try:
            with httpx.Client(timeout=self._timeout_s, transport=self._transport) as http:
                r = http.post(
                    self._URL,
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": self._max_results,
                        "include_raw_content": True,
                    },
                )
                r.raise_for_status()
                payload = r.json()
        except Exception:
            logging.getLogger("forge_mind.research").exception(
                "research provider search failed (fail-closed: no documents)"
            )
            return []
        today = datetime.now(UTC).date().isoformat()
        docs: list[Document] = []
        for item in payload.get("results") or []:
            url = str(item.get("url") or "").strip()
            text = str(item.get("raw_content") or item.get("content") or "").strip()
            if url and text:
                docs.append(
                    Document(
                        url=url,
                        title=str(item.get("title") or url),
                        text=text,
                        fetched_at=today,
                    )
                )
        return docs


def get_research_provider() -> ResearchProviderLike | None:
    """The configured LIVE provider, or None. None -> research is BLOCKED (the toolbelt preflight
    reports it; nothing is faked). ``FORGE_RESEARCH_PROVIDER=tavily`` + ``TAVILY_API_KEY`` wires
    the live search; an unknown/unset provider or a missing key stays honestly None."""
    name = os.environ.get("FORGE_RESEARCH_PROVIDER", "").strip().lower()
    if name == "tavily":
        key = os.environ.get("TAVILY_API_KEY", "").strip()
        if key:
            return TavilyResearchProvider(key)
    return None


__all__ = [
    "Document",
    "FakeResearchProvider",
    "ProposedFact",
    "ResearchProposal",
    "ResearchProviderLike",
    "ResearchResult",
    "TavilyResearchProvider",
    "detect_variants",
    "get_research_provider",
    "propose_research_facts",
    "research_facts",
    "verify_proposed_fact",
]
