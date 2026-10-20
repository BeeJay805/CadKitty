"""Agentic CAD pipeline (Wave 1): Opus authors real build123d code, proven after.

The thesis flip from the bounded DSL path: instead of constraining the LLM to a
box/cylinder typed DSL and deriving geometry in code (high reliability, low
ceiling), we let the model AUTHOR build123d/OpenCASCADE code in a sandbox and
earn trust through a strict validator + a render -> vision-critique -> iterate
loop. Reliability comes from the sandbox + validator + critique loop, not from a
narrow grammar.

Pipeline (Wave 1, single part):
    prompt -> research/DesignBrief -> author (Opus) -> sandbox exec -> validate
           -> render -> vision critique -> fix patch -> (loop) -> acceptance judge

This package is deliberately separable from the bounded DSL path
(``forge_mind.layout`` / ``forge_mind.dsl``) so that path can be retired once the
spike clears its bar.
"""

from __future__ import annotations
