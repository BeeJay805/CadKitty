"""Shared literal count words, not English entailment or unit conversion.

The positive nominal grammar deliberately excludes zero/no. Legacy scalar and
intent readers retain those words. Only the new definite-pair word needs a
count-unit context; it must never manufacture a two-millimetre measurement.
"""

from __future__ import annotations

import re
from types import MappingProxyType

COUNT_WORDS = MappingProxyType(
    {
        "zero": 0,
        "no": 0,
        "single": 1,
        "one": 1,
        "two": 2,
        "both": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twelve": 12,
    }
)
POSITIVE_COUNT_WORDS = MappingProxyType(
    {word: value for word, value in COUNT_WORDS.items() if value > 0}
)
# Unit classification, not a part/request whitelist. These units denote discrete
# multiplicity. Physical units and unknown units cannot acquire a length from both.
COUNT_UNITS = frozenset(
    [
        "count",
        "counts",
        "hole",
        "holes",
        "bore",
        "bores",
        "solid",
        "solids",
        "body",
        "bodies",
        "part",
        "parts",
        "item",
        "items",
        "piece",
        "pieces",
        "screw",
        "screws",
        "bolt",
        "bolts",
        "nut",
        "nuts",
        "fastener",
        "fasteners",
        "instance",
        "instances",
        "occurrence",
        "occurrences",
    ]
)


def count_word_values(quote: str) -> list[float]:
    return [
        float(COUNT_WORDS[word])
        for word in re.findall(r"\b[a-z]+\b", quote.lower())
        if word in COUNT_WORDS
    ]


def source_word_values(quote: str, *, unit: str) -> list[float]:
    """Preserve legacy word values; add 'both' only for discrete count units."""
    return [
        float(COUNT_WORDS[word])
        for word in re.findall(r"\b[a-z]+\b", quote.lower())
        if word in COUNT_WORDS and (word != "both" or unit.casefold().strip() in COUNT_UNITS)
    ]
