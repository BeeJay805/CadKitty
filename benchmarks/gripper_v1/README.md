gripper_v1 — Frozen Benchmark Suite
=====================================

This benchmark tests a text-to-CAD engine's ability to produce correct gripper
and manipulator assemblies driven by an MG996R servo. It contains 5 prompts
(4 in class "gripper", 1 control prompt in class "arm"), each with 2 paraphrases;
a prompt's score is the minimum across all three variants.

Frozen rule: any content change (prompt text, paraphrase, expect value, or schema
key) constitutes a new benchmark version. Rename the directory and version field;
never edit this version. Scores are only meaningful when compared within the same
version and the same model provider.

The CAD author model never sees these files. The rubric was authored by an
independent model (claude-sonnet-4-6) so that expected values reflect what a
correct result must contain, not how the engine was built to produce it.
