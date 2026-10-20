"""HTTP surface for the agentic CAD engine.

A thin FastAPI server that wraps ``forge_mind.agentic.result.produce_part`` and projects its
result into the ``GenerateResponse`` contract the web frontend builds against. The engine,
validator, and renderer are untouched -- this package is pure adapt + serialize.

The FastAPI instance lives at ``forge_mind.api.app:app`` (do not re-export it here -- re-exporting
``app`` into this package namespace would shadow the ``app`` submodule and break that uvicorn target).

Run locally:
    uv run --extra api uvicorn forge_mind.api.app:app --reload --port 8000
"""
