"""Failure-analysis layer for end-to-end pipeline reliability.

Measurement only: this package classifies and aggregates pipeline failures so we
know what to improve first. It does not change any pipeline stage, prompt, or
schema. It runs the existing pipeline, labels each trial's outcome with a failure
taxonomy, and produces a dataset (counts by label, by stage, top failing checks,
cost/latency, links to representative artifacts).
"""
