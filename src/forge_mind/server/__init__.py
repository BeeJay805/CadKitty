"""HTTP server: a thin FastAPI wrapper around ``run_pipeline`` for the web frontend.

Additive only — imports the existing pipeline and maps its result to the JSON shape the
Next.js app expects (``web/src/lib/schema.ts``). It does not touch the core (intent /
blueprint / compiler / validator / kit), so byte-equivalence of generation is unaffected.
"""
