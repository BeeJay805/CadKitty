"""Seeded-defect validator audit (ADR-0083): mutation operators + defect confirmers.

Measures the production verdict's false-PASS / false-FAIL rates by injecting labeled
defects into known-good corpus parts and re-running the SAME sandbox+validate path the
live loop uses. Runner: scripts/audit_validator.py.
"""
