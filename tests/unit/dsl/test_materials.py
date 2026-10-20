"""The FDM filament registry (ADR-0087 N1): cited planning facts, the TPU None-honesty rule,
and registration in the global registry set."""

from __future__ import annotations

from forge_mind.dsl.standards import ALL_REGISTRIES, FILAMENTS, filament


def test_registered_and_all_provisional() -> None:
    assert ALL_REGISTRIES["filament"] is FILAMENTS
    for row in FILAMENTS.all():
        assert row.provisional is True  # cited, NOT physically verified — surfaced, never hidden
        assert row.source and len(row.source) > 20  # every value traces (L2 binds the implementer)


def test_cited_heat_ordering_holds() -> None:
    # the planning lever: the conservative cited HDT ordering (PLA lowest of the rigid set)
    by = {r.key: r for r in FILAMENTS.all()}
    assert by["PLA"].hdt_045_c == 55.0
    assert by["PETG"].hdt_045_c == 70.0
    assert by["ABS"].hdt_045_c == 95.0
    assert by["ASA"].hdt_045_c == 95.0
    assert by["Nylon"].hdt_045_c == 90.0
    rigid = [by[k].hdt_045_c or 0 for k in ("PLA", "PETG", "Nylon", "ABS")]
    assert rigid == sorted(rigid)


def test_tpu_heat_fields_are_honestly_none() -> None:
    # Simplify3D marks TPU 95A Tg "n/a (rubbery)"; no published HDT -> a number would be INVENTED
    tpu = filament("TPU")
    assert tpu.tg_c is None and tpu.hdt_045_c is None
    assert tpu.flexible is True
