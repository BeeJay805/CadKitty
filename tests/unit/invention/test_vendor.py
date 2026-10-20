"""The vendor-research gate (ADR-0048): confirm-or-FLAG with a `listed` tier that NEVER equals the
dimensional `verified`. The adversarial cases are the point — a wrong or missing listing must be
FLAGGED, never admitted as a buyable line (the highest false-PASS surface the pivot opens)."""

from __future__ import annotations

from forge_mind.agentic.provenance import Tier
from forge_mind.invention import (
    NullFetcher,
    PartRequirement,
    VendorListing,
    VendorProvenance,
    research_vendor_parts,
)


class _StubFetcher:
    """Returns canned listings per query and records what it was asked."""

    def __init__(self, by_query: dict[str, list[VendorListing]]) -> None:
        self.by_query = by_query
        self.queries: list[str] = []

    def search(self, query: str) -> list[VendorListing]:
        self.queries.append(query)
        return self.by_query.get(query, [])


def _req(designation: str = "M3x12 SHCS", query: str = "m3x12 shcs") -> PartRequirement:
    return PartRequirement(designation=designation, category="screw", query=query)


def _listing(designation: str, sku: str = "91290A115") -> VendorListing:
    return VendorListing(
        designation=designation,
        sku=sku,
        title=f"{designation} socket head cap screw",
        url="https://example.test/p/" + sku,
        vendor="ExampleCo",
        price_usd=0.12,
    )


def test_null_fetcher_flags_everything_fail_closed() -> None:
    """With no real catalog wired, nothing is confirmed — every requirement FLAGs, zero lines.
    Absence of a fetcher means absence of a SKU; the gate can never invent one."""
    bom = research_vendor_parts([_req(), _req("608 bearing", "608 bearing")], NullFetcher())
    assert bom.lines == []
    assert set(bom.flagged) == {"M3x12 SHCS", "608 bearing"}
    assert bom.all_confirmed is False


def test_exact_match_admits_a_listed_line() -> None:
    fetcher = _StubFetcher({"m3x12 shcs": [_listing("M3x12 SHCS")]})
    bom = research_vendor_parts([_req()], fetcher)
    assert len(bom.lines) == 1
    line = bom.lines[0]
    assert line.provenance is VendorProvenance.LISTED
    assert (
        line.listing.sku == "91290A115"
    )  # the SKU comes from the FETCHED listing, never the model
    assert line.listing.price_usd == 0.12
    assert bom.flagged == []
    assert bom.all_confirmed is True


def test_a_different_part_is_flagged_never_admitted() -> None:
    """THE moat-critical case: the fetcher returns a real listing for the WRONG part (M4, not M3).
    The gate checks the returned designation and FLAGs — it does not admit a plausible wrong SKU."""
    fetcher = _StubFetcher({"m3x12 shcs": [_listing("M4x12 SHCS", sku="WRONG")]})
    bom = research_vendor_parts([_req()], fetcher)
    assert bom.lines == []
    assert bom.flagged == ["M3x12 SHCS"]


def test_empty_result_is_flagged() -> None:
    bom = research_vendor_parts([_req()], _StubFetcher({}))
    assert bom.lines == []
    assert bom.flagged == ["M3x12 SHCS"]


def test_partial_confirmation_keeps_the_unconfirmed_flagged() -> None:
    fetcher = _StubFetcher(
        {
            "m3x12 shcs": [_listing("M3x12 SHCS")],
            "608 bearing": [],  # not found -> must flag, never a line
        }
    )
    bom = research_vendor_parts(
        [_req(), _req("608 bearing", "608 bearing")],
        fetcher,
    )
    assert [line.designation for line in bom.lines] == ["M3x12 SHCS"]
    assert bom.flagged == ["608 bearing"]
    assert bom.all_confirmed is False


def test_match_is_case_and_whitespace_insensitive_but_not_fuzzy() -> None:
    # normalization tolerates case + spacing...
    fetcher = _StubFetcher({"q": [_listing("  m3x12   shcs ")]})
    bom = research_vendor_parts([_req(query="q")], fetcher)
    assert len(bom.lines) == 1
    # ...but a substring/near-miss is NOT a match (no fuzzy admission).
    near = _StubFetcher({"q": [_listing("M3x12 SHCS low-profile")]})
    assert research_vendor_parts([_req(query="q")], near).flagged == ["M3x12 SHCS"]


def test_listed_is_a_separate_axis_never_verified() -> None:
    """A vendor offer is volatile; there is no path from LISTED to the dimensional `verified` tier."""
    assert {v.value for v in VendorProvenance} == {"listed", "unconfirmed"}
    assert "verified" not in {v.value for v in VendorProvenance}
    assert Tier.VERIFIED.value == "verified"  # the dimensional axis is distinct and untouched


def test_fetcher_is_searched_with_the_requirement_query() -> None:
    fetcher = _StubFetcher({"m3x12 shcs": [_listing("M3x12 SHCS")]})
    research_vendor_parts([_req(query="m3x12 shcs")], fetcher)
    assert fetcher.queries == ["m3x12 shcs"]
