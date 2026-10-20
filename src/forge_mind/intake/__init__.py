"""Planner-first design intake (the layer BEFORE CAD generation).

A non-trivial request should not jump straight from prompt to CAD. It first flows through a
*Planner Agent* that understands the request (text + optional image/drawing + optional mouse
sketch), decides what is already known vs. missing, asks only the questions that affect geometry /
fit / interfaces / fasteners / dimensions / printability, and emits a structured ``DesignBrief``
the EXISTING pipeline consumes.

This package is ADDITIVE and engine-agnostic: it sits above ``forge_mind.pipeline.run_pipeline``
and never touches the compiler or validator. Named ``intake`` (not ``planner``) because
``forge_mind.planner`` already exists as the multi-part *decomposition* planner (ADR-0021) -- a
different concern.

Public surface:
    schema   -- typed Question / Answer / DesignBrief / Asset / Sketch models + enums
    session  -- DesignSession + a pluggable SessionStore (in-memory / JSON file)
    agent    -- PlannerAgent + a mockable PlannerBrain (deterministic Heuristic default, optional LLM)
    service  -- IntakeService: the orchestration the API (or a notebook) drives
    brief    -- DesignBrief readiness gate + brief -> prompt wiring into the existing pipeline

The HTTP layer lives in ``forge_mind.intake.api`` and is imported separately (it needs the optional
``fastapi`` extra), so importing this package stays dependency-light.
"""

from __future__ import annotations

from forge_mind.intake.schema import (
    Answer,
    Asset,
    AssetType,
    CriticalDimension,
    DesignBrief,
    DimensionSource,
    KnownStandard,
    Option,
    Question,
    QuestionType,
    RequiredInterface,
    SessionStatus,
    Sketch,
    SketchAnnotation,
    SketchAsset,
    SketchPoint,
    SketchStroke,
)
from forge_mind.intake.session import DesignSession, InMemorySessionStore, SessionStore

__all__ = [
    "Answer",
    "Asset",
    "AssetType",
    "CriticalDimension",
    "DesignBrief",
    "DesignSession",
    "DimensionSource",
    "InMemorySessionStore",
    "KnownStandard",
    "Option",
    "Question",
    "QuestionType",
    "RequiredInterface",
    "SessionStatus",
    "SessionStore",
    "Sketch",
    "SketchAnnotation",
    "SketchAsset",
    "SketchPoint",
    "SketchStroke",
]
