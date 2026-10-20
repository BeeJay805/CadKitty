"""Clarifying questions — the lab's intake step (problem -> questions -> answers -> grounded context).

The front door turns a PROBLEM into concepts, but a bare problem under-specifies the build (envelope,
mounting, material, load, quantity). This is the missing middle: ONE brain pass proposes a few
high-value clarifying questions; the answers (a human's, or a simulated maker's in an unattended test)
fold into a STATED-FACTS block that grounds BOTH the front door's concepts and the engine's brief
research (via ``produce_part``'s ``extra_context``, ADR-0036). The ``client`` seam is injectable, so the
whole step is offline-testable with no spend.

Honesty: the questions never assert facts and never gate; an unanswered question is simply absent from
the context (nothing is invented here). Default model is :data:`BRAIN_MODEL` (Opus today; ADR-0055)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.config import BRAIN_MODEL
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_MAX_TOKENS = 2000


class ClarifyingQuestion(BaseModel):
    """One high-value question that pins a build constraint the bare problem left open."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(..., description="The question to ask the user, plainly.")
    pins: str = Field(
        ...,
        description="The build constraint it pins: 'envelope' | 'mounting/interface' | "
        "'material/environment' | 'load/use' | 'quantity' (or similar).",
    )
    why: str = Field(..., description="One sentence: why the answer changes the design.")


class ClarificationSet(BaseModel):
    """The brain's clarifying questions for one problem (structured-output container)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: list[ClarifyingQuestion]


class AnsweredQuestion(BaseModel):
    """A question paired with the user's (or simulated maker's) concrete answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    answer: str


class AnswerSet(BaseModel):
    """The answers to a clarification set (structured-output container for the test user-sim)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answers: list[AnsweredQuestion]


_QUESTION_SYSTEM = """You are the intake engineer of a physical-invention lab. A user brings a PROBLEM. \
Before any concept or CAD work, ask a FEW high-value clarifying questions -- only ones whose answers \
would actually change the mechanism, the geometry, or the materials. Pin CONCRETE build constraints a \
maker must know: size/envelope, what it mounts to or interfaces with, material + environment \
(indoor/outdoor, heat, load), the load or use cycle, and quantity. Ask 3-4 questions, each pinning a \
DIFFERENT constraint. Do NOT ask vague or preference questions, and do NOT propose a solution yet. For \
each: the question, the constraint it pins, and one sentence on why it matters."""

_ANSWER_SYSTEM = """You are a typical maker who brought a problem to an invention lab and is answering \
the intake engineer's clarifying questions. Answer each PLAUSIBLY and CONCRETELY for a normal home/shop \
context: give specific numbers (mm, kg, counts) and materials where asked, and keep each answer to a \
sentence or two. You are simulating a real user so the lab can be tested end-to-end -- answer as that \
user would, never refuse."""


def generate_clarifying_questions(
    problem: str,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    max_tokens: int = _MAX_TOKENS,
) -> tuple[list[ClarifyingQuestion], LLMUsage]:
    """One brain pass: PROBLEM -> a few high-value clarifying questions. ``client`` is injectable
    (mock in tests). ``model`` defaults to :data:`BRAIN_MODEL` (Opus today; ``FORGE_BRAIN_MODEL`` swaps)."""
    client = client or make_client()
    user = f"Problem:\n{problem.strip()}\n\nAsk your 3-4 clarifying questions."
    result, usage = client.generate_structured(
        system=_QUESTION_SYSTEM,
        user=user,
        output_model=ClarificationSet,
        model=model or BRAIN_MODEL,
        max_tokens=max_tokens,
    )
    return list(result.questions), usage


def answer_clarifying_questions(
    problem: str,
    questions: list[ClarifyingQuestion],
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    max_tokens: int = _MAX_TOKENS,
) -> tuple[list[AnsweredQuestion], LLMUsage]:
    """Simulate a maker answering the questions -- ONLY so the pipeline can run end-to-end UNATTENDED in
    a test. A real run takes the human's answers instead; this stand-in is never used in production."""
    client = client or make_client()
    qlist = "\n".join(f"{i + 1}. {q.question}" for i, q in enumerate(questions))
    user = (
        f"Problem:\n{problem.strip()}\n\nThe intake engineer asks:\n{qlist}\n\n"
        "Answer each one concretely (numbers + materials where relevant)."
    )
    result, usage = client.generate_structured(
        system=_ANSWER_SYSTEM,
        user=user,
        output_model=AnswerSet,
        model=model or BRAIN_MODEL,
        max_tokens=max_tokens,
    )
    return list(result.answers), usage


def clarifications_to_context(answers: list[AnsweredQuestion]) -> str:
    """Format answered questions as a STATED-FACTS block (ADR-0036 shape) to ground the front door and
    the engine brief. Deterministic; the facts are the user's, never invented here. Empty -> ''."""
    if not answers:
        return ""
    lines = ["STATED FACTS (the user's answers to clarifying questions -- treat as given):"]
    lines += [f"- {a.question.strip()} -> {a.answer.strip()}" for a in answers]
    return "\n".join(lines)


__all__ = [
    "AnswerSet",
    "AnsweredQuestion",
    "ClarificationSet",
    "ClarifyingQuestion",
    "answer_clarifying_questions",
    "clarifications_to_context",
    "generate_clarifying_questions",
]
