"""Manufacturer tuple values are literal evidence, not semantic confirmation."""

import pytest

from forge_mind.workspace.product_sources import quoted_numbers


@pytest.mark.parametrize(
    ("quote", "value"),
    [
        ("120x120x25 mm", 25),
        ("120X120X25 mm", 25),
        ("120 \u00d7 120 \u00d7 25 mm", 25),
        ("120x120x25mm", 25),
        ("120X120X25MM", 25),
        ("12.5 x .8 cm", 0.8),
        ("1.2e2x1.2e2x2.5e1 mm", 25),
        ("Thickness 25mm", 25),
        ("Offset -25 mm", -25),
        ("3 x -25 mm", -25),
        ("Offset \u221225 mm", -25),
        ("1e\u22123 mm", 0.001),
        ("Fan speed 1700rpm", 1700),
        ("Flow 102.1m3/h", 102.1),
        ("Thickness 25+/-0.1 mm", 25),
        ("Thickness 25+/-0.1 mm", 0.1),
        ("Thickness 25\u00b10.1 mm", 25),
        ("25-26 mm", 26),
        ("-25--20 C", -20),
    ],
)
def test_literal_dimension_components_are_available_without_conversion(quote, value):
    assert value in quoted_numbers(quote)


@pytest.mark.parametrize(
    ("quote", "value"),
    [
        ("NF-A12x25 PWM", 25),
        ("M3x25 screw", 25),
        ("120x120x25ABC", 25),
        ("120x120x25ABC", 120),
        ("25ABC", 25),
        ("1e+SKU", 1),
        ("\u221225 mm", 25),
        ("3x\u221225 mm", 25),
        ("Thickness 25+/-0.1 mm", -0.1),
        ("25-26ABC", 26),
        ("SKU-25-26", 26),
        ("120x120x250 mm", 25),
        ("120x120x25.5 mm", 25),
        ("3x-25 mm", 25),
        ("A-25", 25),
        ("12.5 x .8 cm", 8),
        ("120x120x25 mm", 210),
    ],
)
def test_tuple_support_does_not_invent_substrings_units_or_sku_components(quote, value):
    assert value not in quoted_numbers(quote)
