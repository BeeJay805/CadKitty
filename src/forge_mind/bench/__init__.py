"""Benchmark harness — runs a benchmark as N measured trials and reports.

Harness CODE lives here (in the package, so it's importable, mypy-checked, and
unit-testable). Benchmark DATA (prompts, ground truth, pass criteria, results)
lives under the top-level ``benchmarks/`` directory.

The harness is runner-agnostic: a Runner maps a prompt to a TrialOutput. Phase 3
ships the HandwrittenRunner (bypasses the LLM, feeds the reference DSL straight
to compile+validate) so the measurement infrastructure exists before any LLM
stage does. Later phases add a PipelineRunner (real LLM) and a BaselineRunner
(naive single-call) without touching the harness core.
"""
