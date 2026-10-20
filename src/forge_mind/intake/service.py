"""IntakeService -- the orchestration the API (or a notebook/CLI) drives.

It wires the store + planner agent + asset handling into the guided flow: create a session, add input
(text / image / sketch), run the planner, answer questions, finalize a brief, and -- only once the
brief is ready -- start the EXISTING generation pipeline. The engine is never modified: generation is
a ``brief -> precise prompt -> run_pipeline`` handoff, injected as ``generate_fn`` so the service is
testable without the CAD stack.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from forge_mind.intake.agent import PlannerAgent, PlannerOutcome
from forge_mind.intake.assets import save_image_asset, save_sketch_asset
from forge_mind.intake.brief import brief_to_prompt, readiness
from forge_mind.intake.schema import (
    Answer,
    Asset,
    AssetType,
    CaptureContext,
    DesignBrief,
    SessionStatus,
    Sketch,
    SketchAsset,
)
from forge_mind.intake.session import (
    DesignSession,
    InMemorySessionStore,
    SessionNotFoundError,
    SessionStore,
    new_session_id,
)

# A generation backend: (prompt, output_dir) -> some result object. Default lazily wraps run_pipeline.
GenerateFn = Callable[[str, Path], object]


class PlannerNotReadyError(RuntimeError):
    """Finalize was requested but the planner still has required questions / a refusal."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons) or "the planner is not ready")


class GenerationBlockedError(RuntimeError):
    """Generation was requested but the brief is not ready / required answers are missing."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons) or "generation is blocked")


def _default_generate(prompt: str, output_dir: Path) -> object:
    """The real backend: feed the synthesized prompt to the existing pipeline. Imported lazily so the
    service (and the API) don't pull the heavy CAD stack unless a generation actually runs."""
    from forge_mind.pipeline import run_pipeline

    return run_pipeline(prompt, output_dir, use_layoutspec=True)


class IntakeService:
    def __init__(
        self,
        *,
        store: SessionStore | None = None,
        agent: PlannerAgent | None = None,
        asset_dir: Path | str = "benchmarks/results/_intake",
        generate_fn: GenerateFn | None = None,
    ) -> None:
        self.store: SessionStore = store or InMemorySessionStore()
        self.agent = agent or PlannerAgent()
        self.asset_dir = Path(asset_dir)
        self._generate_fn = generate_fn or _default_generate

    # -- session lifecycle ------------------------------------------------------------------------

    def create_session(self) -> DesignSession:
        return self.store.create()

    def get_session(self, session_id: str) -> DesignSession:
        session = self.store.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    # -- input ------------------------------------------------------------------------------------

    def add_input(
        self,
        session_id: str,
        *,
        text: str | None = None,
        sketch: Sketch | None = None,
        capture: CaptureContext | None = None,
        base_url: str | None = None,
        printable: bool | None = None,
    ) -> DesignSession:
        """Add a text prompt, a Draw it! sketch, and/or the multimodal capture bundle (speech, scale,
        viewpoint, notes, labels -- ADR-0025). Image uploads go through ``add_asset``.

        ``printable`` (the ADR-0031 toggle, ADR-0036) sets the session's fabrication intent:
        True -> "print_3d", False -> "general", None (not sent) -> unchanged (legacy behavior)."""
        session = self.get_session(session_id)
        if printable is not None:
            session.fabrication = "print_3d" if printable else "general"
        if text is not None:
            session.original_text_prompt = text
        if sketch is not None:
            self._attach_sketch(session, sketch, base_url=base_url)
        if capture is not None:
            session.capture = capture
            if capture.scale is not None:
                # A declared in-plane scale IS a real dimension -> the drawing is no longer pixels-only.
                for asset in session.uploaded_assets:
                    asset.scale_confirmed = True
                for sk in session.sketch_assets:
                    sk.scale_confirmed = True
        session.status = SessionStatus.COLLECTING_INPUT
        self.store.save(session)
        return session

    def add_asset(
        self,
        session_id: str,
        *,
        filename: str,
        data: bytes,
        asset_type: AssetType = AssetType.REFERENCE_IMAGE,
        content_type: str = "application/octet-stream",
        base_url: str | None = None,
    ) -> Asset:
        session = self.get_session(session_id)
        asset = save_image_asset(
            self.asset_dir / session.id,
            asset_id=new_session_id(),
            filename=filename,
            data=data,
            asset_type=asset_type,
            content_type=content_type,
            base_url=base_url,
        )
        session.uploaded_assets.append(asset)
        self.store.save(session)
        return asset

    def add_sketch(
        self, session_id: str, *, sketch: Sketch, base_url: str | None = None
    ) -> SketchAsset:
        session = self.get_session(session_id)
        sketch_asset = self._attach_sketch(session, sketch, base_url=base_url)
        self.store.save(session)
        return sketch_asset

    def _attach_sketch(
        self, session: DesignSession, sketch: Sketch, *, base_url: str | None = None
    ) -> SketchAsset:
        sketch_asset = save_sketch_asset(
            self.asset_dir / session.id, asset_id=new_session_id(), sketch=sketch, base_url=base_url
        )
        session.sketch_assets.append(sketch_asset)
        return sketch_asset

    def asset_file_path(self, session_id: str, name: str) -> Path | None:
        """Resolve a stored asset / sketch-preview file for serving, guarding path traversal. Returns
        None for an unknown or escaping name (the API turns that into a 404)."""
        session = self.get_session(session_id)
        base = (self.asset_dir / session.id).resolve()
        candidate = (base / name).resolve()
        if candidate.parent != base or not candidate.is_file():
            return None
        return candidate

    # -- planner ----------------------------------------------------------------------------------

    def run_planner(self, session_id: str) -> PlannerOutcome:
        """Run a planner pass; persist the questions/state/brief and update the session status."""
        session = self.get_session(session_id)
        outcome = self.agent.run(session)
        session.planner_state = outcome.state
        session.questions = outcome.questions
        if outcome.ready and outcome.brief is not None:
            session.final_design_brief = outcome.brief
            session.status = SessionStatus.READY_FOR_BLUEPRINT
        else:
            # A refusal or outstanding questions both keep us collecting clarification (never generate).
            session.final_design_brief = None
            session.status = SessionStatus.AWAITING_CLARIFICATION
        self.store.save(session)
        return outcome

    def answer_questions(self, session_id: str, answers: Iterable[Answer]) -> DesignSession:
        session = self.get_session(session_id)
        for answer in answers:
            session.answers.append(answer)
            # Supplying the part's real width confirms the drawing's scale (it is no longer pixels-only).
            if answer.question_id == "q_image_scale":
                for asset in session.uploaded_assets:
                    asset.scale_confirmed = True
                for sk in session.sketch_assets:
                    sk.scale_confirmed = True
        self.store.save(session)
        return session

    def finalize(self, session_id: str) -> DesignBrief:
        """Produce the final DesignBrief, or raise if the planner still needs answers."""
        outcome = self.run_planner(session_id)
        if outcome.refusal:
            raise PlannerNotReadyError([outcome.refusal])
        if not outcome.ready or outcome.brief is None:
            raise PlannerNotReadyError([q.id for q in outcome.questions] or ["not ready"])
        return outcome.brief

    # -- generation (gated handoff to the existing pipeline) --------------------------------------

    def start_generation(self, session_id: str) -> object:
        """Start the existing CAD pipeline from the finalized brief -- BLOCKED unless ready.

        Gate: a ready brief, no unanswered required questions, no UNKNOWN-source critical dimension.
        Then the brief is rendered to a precise prompt and handed to ``generate_fn``; the engine,
        validator, and kit are untouched.
        """
        session = self.get_session(session_id)
        ok, reasons = readiness(session)
        if not ok or session.final_design_brief is None:
            raise GenerationBlockedError(reasons or ["no ready design brief"])

        prompt = brief_to_prompt(session.final_design_brief)
        generation_id = new_session_id()
        session.linked_generation_id = generation_id
        session.status = SessionStatus.GENERATING
        self.store.save(session)

        job_dir = self.asset_dir / session.id / "generation" / generation_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = self._generate_fn(prompt, job_dir)
        except Exception:
            session.status = SessionStatus.FAILED
            self.store.save(session)
            raise
        session.status = SessionStatus.COMPLETE
        self.store.save(session)
        return result

    def generation_prompt(self, session_id: str) -> str:
        """The exact prompt that would be sent to the pipeline (handy for review/debugging)."""
        session = self.get_session(session_id)
        if session.final_design_brief is None:
            raise GenerationBlockedError(["no ready design brief"])
        return brief_to_prompt(session.final_design_brief)


__all__ = [
    "GenerateFn",
    "GenerationBlockedError",
    "IntakeService",
    "PlannerNotReadyError",
    "SessionNotFoundError",
]
