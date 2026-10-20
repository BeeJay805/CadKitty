"""Pure controller and immutable-target controls. Fake STEP is never CAD proof."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from tests.unit.workspace.test_photo_nominal_claims import SOURCE, wire

from forge_mind.workspace import photo_generation as generation
from forge_mind.workspace import photo_profile_service as profiles
from forge_mind.workspace.document import digest
from forge_mind.workspace.photo_profile import _rectangle_fit, analyze_profile
from forge_mind.workspace.source_delivery_contract_draft import preflight_source_delivery_draft
from forge_mind.workspace.store import Store, WorkspaceError


def photo_v2_wire(proposal=None):
    """Represent the same retained source duties in the prospective typed wire."""
    raw = (proposal or wire()).model_dump(mode="json")
    claims = raw.pop("photo_nominal_claims")
    raw.pop("nominal_targets")
    supported = {row["requirement_id"]: row for row in claims if row["claims"]}
    raw["photo_requirements"] = [
        {
            **{key: row[key] for key in ("id", "source", "critical", "quote_ref", "subject_ids")},
            "facets": supported[row["id"]]["claims"],
            **{
                key: supported[row["id"]][key]
                for key in ("size_fact_ids", "hole_count_fact_id", "frame_policy")
            },
        }
        for row in raw["requirements"]
        if row["id"] in supported
    ]
    raw["requirements"] = [row for row in raw["requirements"] if row["id"] not in supported]
    return raw


def test_no_photo_boundaries_use_existing_authenticated_get_protocol_only():
    reads = []

    def get(sid, owner):
        reads.append((sid, owner))
        if owner != "owner":
            raise WorkspaceError("workspace not found")
        return {"authoring_mode": "native_operations"}

    service = SimpleNamespace(store=SimpleNamespace(get=get))
    assert generation.generation_context(service, "session", "owner", None, None) is None
    generation.require_grounded(service, "session", "owner")
    assert generation.photo_closure(service, "session", "owner", None) == {
        "status": "not_applicable"
    }
    assert generation._read(service, "session", "owner") is None
    assert reads == [("session", "owner")] * 4
    with pytest.raises(WorkspaceError, match="not found"):
        generation.photo_closure(service, "session", "other", None)


def test_photo_source_with_valid_claims_corrects_original_wire_not_canonical(tmp_path, monkeypatch):
    from tests.unit.workspace.test_requirement_verification import geometry_target

    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Requirement
    from forge_mind.workspace.source_delivery_contract_draft import PhotoSourceContractDraftV2

    initial = photo_v2_wire()
    initial["canonical_requirements"].append(
        {
            "requirement": Requirement(
                id="width_check",
                text="Copy the visible plate",
                source_quote=SOURCE,
                verification=geometry_target("nominal_perforated_box").verification.model_copy(
                    update={"source_quote": SOURCE}
                ),
            ).model_dump(mode="json"),
            "subject_ids": ["plate"],
        }
    )
    initial["requirement_order"].append("width_check")
    corrected = deepcopy(initial)
    corrected["canonical_requirements"][-1]["requirement"]["verification"] = None
    assert (
        preflight_source_delivery_draft(
            PhotoSourceContractDraftV2.model_validate(initial), SOURCE
        ).compilation["status"]
        == "invalid_interpretation"
    )
    assert (
        preflight_source_delivery_draft(
            PhotoSourceContractDraftV2.model_validate(corrected), SOURCE
        ).compilation["status"]
        == "ready"
    )
    image = Image.new("RGB", (32, 24), "white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "d" * 32,
            "prompt": SOURCE,
            "reference": {
                "filename": "source.png",
                "media_type": "image/png",
                "content_base64": base64.b64encode(stream.getvalue()).decode(),
            },
        },
    )
    calls = []

    class Client:
        def generate_structured_with_image(self, **kwargs):
            calls.append(kwargs)
            assert len(calls) <= 2
            assert kwargs["output_model"] is PhotoSourceContractDraftV2
            return kwargs["output_model"].model_validate(
                initial if len(calls) == 1 else corrected
            ), LLMUsage("scripted", 0, 0)

    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    service._run_agent_job({**store.claim(), "max_actions": 0, "max_seconds": 60})
    saved = store.get(session["id"], "owner")
    assert len(calls) == 2 and calls[0]["image_png"] == calls[1]["image_png"]
    assert saved["intent_compilation"]["status"] == "ready"
    assert (
        saved["intent_compilation"]["source_proposal"]["raw_response"]["photo_requirements"]
        == PhotoSourceContractDraftV2.model_validate(corrected).model_dump(mode="json")[
            "photo_requirements"
        ]
    )
    assert not saved["revisions"] and not saved.get("accepted_revision")


def test_fresh_photo_typed_duties_admit_actual_failed_prose_without_using_it(tmp_path, monkeypatch):
    from tests.unit.workspace.test_photo_nominal_claims import claim, draft

    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.customer_references import reference_image
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Requirement
    from forge_mind.workspace.source_delivery_contract_draft import PhotoSourceContractDraftV2

    source = SOURCE.replace("Both holes", "Both visible holes")
    original = draft()
    facets = [
        ["visible_profile"],
        ["rectangular_outline"],
        ["extent_u"],
        ["extent_v"],
        ["thickness"],
        ["opening_count"],
        ["straight_through"],
    ]
    quotes = [
        source.split(". Both")[0] + ".",
        source.split(". Both")[0] + ".",
        source.split(". Both")[0] + ".",
        source.split(". Both")[0] + ".",
        source.split(". Both")[0] + ".",
        "Both visible holes go straight through.",
        "Both visible holes go straight through.",
    ]
    original.contract.requirements = [
        Requirement(
            id=f"r{index}",
            source_quote=quote,
            text="Copy the plate shown in the supplied reference, including its visible profile and hole arrangement.",
        )
        for index, quote in enumerate(quotes, 1)
    ]
    original.attribution.requirement_subjects = {f"r{i}": ["plate"] for i in range(1, 8)}
    for fact, ids in zip(
        original.attribution.facts,
        [["r3", "r4"], ["r3", "r4"], ["r5"], ["r6", "r7"]],
        strict=True,
    ):
        fact.requirement_ids = ids
        fact.source_quote = fact.source_quote.replace("Both holes", "Both visible holes")
    legacy = wire(
        original, [claim(requirement_id=f"r{i}", claims=value) for i, value in enumerate(facets, 1)]
    )
    legacy.photo_reference.source_quote = source
    legacy_before = legacy.model_dump(mode="json")
    rejected = preflight_source_delivery_draft(legacy, source)
    assert rejected.compilation["status"] == "invalid_interpretation"
    assert legacy.model_dump(mode="json") == legacy_before
    raw = photo_v2_wire(legacy)
    raw["photo_reference"]["source_quote"] = source
    proposal = PhotoSourceContractDraftV2.model_validate(raw)
    assert all("text" not in row for row in raw["photo_requirements"])
    preflight = preflight_source_delivery_draft(proposal, source)
    assert preflight.compilation["status"] == "ready", preflight.compilation
    assert [row.id for row in preflight.draft.contract.requirements] == [
        f"r{i}" for i in range(1, 8)
    ]
    assert all(row.critical and row.verification for row in preflight.draft.contract.requirements)
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(stream, format="PNG")
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "e" * 32,
            "prompt": source,
            "reference": {
                "filename": "source.png",
                "media_type": "image/png",
                "content_base64": base64.b64encode(stream.getvalue()).decode(),
            },
        },
    )
    calls = []

    class Client:
        def generate_structured_with_image(self, **kwargs):
            calls.append(kwargs)
            assert len(calls) == 1 and kwargs["output_model"] is PhotoSourceContractDraftV2
            assert "photo_nominal_claims MUST" not in kwargs["system"]
            return proposal, LLMUsage("scripted-v2-photo", 0, 0)

    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    service._run_agent_job({**store.claim(), "max_actions": 0, "max_seconds": 60})
    saved = store.get(session["id"], "owner")
    assert len(calls) == len(saved["usage"]) == 1
    assert calls[0]["image_png"] == reference_image(store, session["id"], "owner")[0]
    assert saved["intent_compilation"]["status"] == "ready"
    assert (
        saved["intent_compilation"]["source_proposal"]["protocol"]
        == "photo_source_contract_draft_v2"
    )
    assert saved["customer_reference"]["known_dimension"] is None
    assert not saved["revisions"] and not saved.get("accepted_revision")


@pytest.fixture
def photo(tmp_path, monkeypatch):
    # This is a fake worker contract, not qualification of concurrently edited source.
    monkeypatch.setattr(profiles, "engine_fingerprint", lambda: "e" * 64)
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 80, 570, 380), fill="black")
    draw.ellipse((160, 150, 220, 210), fill="white")
    draw.ellipse((390, 260, 470, 340), fill="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    preflight = preflight_source_delivery_draft(wire(), SOURCE)
    assert preflight.compilation["status"] == "ready"
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "a" * 32,
            "prompt": SOURCE,
            "contract": preflight.draft.contract.model_dump(mode="json"),
            "reference": {
                "filename": "source.png",
                "media_type": "image/png",
                "content_base64": base64.b64encode(stream.getvalue()).decode(),
            },
        },
    )
    job = store.claim()
    if job.get("input_application_id"):
        store.apply_input_contract(
            session["id"], "owner", job["id"], job["input_application_id"], preflight.draft.contract
        )
    store.update(
        session["id"],
        intent_compilation={
            **preflight.compilation,
            "job_id": job["id"],
            "source_hash": digest(SOURCE),
            "source_proposal": preflight.source_proposal,
        },
    )
    calls = []

    def worker(directory, payload, timeout, cancelled):
        assert not cancelled() and 0 < timeout <= 30
        calls.append(payload["operation"])
        if payload["operation"] == "photo_profile_analyze":
            return analyze_profile((directory / "vision.png").read_bytes(), payload["request"])
        data = (directory / "part.step").read_bytes()
        if payload["operation"] == "nominal_photo_solid":
            request = payload["request"]
            return {
                "status": "pass" if data == b"correct fake STEP" else "fail",
                "method": "nominal_photo_solid_v1",
                "criterion_hash": request["criterion_hash"],
                "reason": "Fake whole material witness",
                "facts": {
                    "step_sha256": hashlib.sha256(data).hexdigest(),
                    "target_sha256": request["target_sha256"],
                    "expected_document_sha256": digest(request["expected_document"]),
                    "source_constraints": request["source_constraints"],
                },
                "evidence_refs": [
                    {"artifact": "part.step", "sha256": hashlib.sha256(data).hexdigest()}
                ],
            }
        result = {
            "step_sha256": hashlib.sha256(data).hexdigest(),
            "observation_sha256": payload["observation"]["observation_sha256"],
            "vision_image_sha256": payload["observation"]["vision_image_sha256"],
            "engine_fingerprint": "e" * 64,
            "selector": payload["selector"],
            "body_count": 1,
            "saved_inner_loop_count": 2,
            "observed_inner_loop_count": 2,
            "symmetric_residual_interval_mm": [0, 0.02]
            if data in {b"correct fake STEP", b"hidden pocket fake STEP"}
            else [0.9, 0.92],
            "coverage": {
                "all_selected_face_wires": True,
                "all_wire_edges_supported": True,
                "camera_refitted": False,
            },
            "repair_guidance": [{"location_plane_mm": [20, 10]}],
        }
        result["comparison_sha256"] = profiles._digest(result)
        return result

    monkeypatch.setattr(profiles, "process_revision", worker)
    service = SimpleNamespace(
        store=store, executor=SimpleNamespace(mode="trusted_local"), stop=threading.Event()
    )
    from forge_mind.workspace.edit_service import read_edit_basis
    from forge_mind.workspace.intent_attribution_service import record_proposal

    record_proposal(
        store,
        session["id"],
        "owner",
        job["id"],
        preflight.draft.attribution,
        preflight.draft.contract,
        SOURCE,
        read_edit_basis(service, session["id"], "owner"),
    )
    return SimpleNamespace(
        service=service,
        store=store,
        sid=session["id"],
        job=job,
        contract=preflight.draft.contract,
        calls=calls,
    )


def grounding(photo):
    generation.observe_inventory(
        photo.service, photo.sid, "owner", photo.job["id"], timeout=30, cancelled=lambda: False
    )
    context = generation.generation_context(photo.service, photo.sid, "owner", None, photo.contract)
    inventory = next(
        item for item in context["inventory"] if item["request"]["foreground"] == "dark"
    )
    component = next(
        c for c in inventory["result"]["components"] if not c["touches_region_boundary"]
    )
    vertices = component["loops"][0]["detected_features"]["vertices"]
    # Choose image TL/TR/BR/BL from detected intersections, never hidden truth.
    ids = [
        min(vertices, key=lambda p: (p["pixel"][0] - x) ** 2 + (p["pixel"][1] - y) ** 2)["id"]
        for x, y in [(70, 80), (570, 80), (570, 380), (70, 380)]
    ]
    return generation.PhotoGrounding(
        inventory_observation_id=inventory["id"],
        component_id=component["id"],
        subject_id="plate",
        part_id="part",
        corner_feature_ids=ids,
        size_fact_ids=["width", "height", "depth"],
        hole_count_fact_id="count",
        planar_hypothesis=True,
        negligible_distortion_hypothesis=True,
    )


def freeze(photo):
    return generation.ground_target(
        photo.service,
        photo.sid,
        "owner",
        photo.job["id"],
        grounding(photo),
        photo.contract,
        timeout=30,
        cancelled=lambda: False,
    )


def revision(photo, target, rid, data):
    from forge_mind.workspace.models import DesignPlan
    from forge_mind.workspace.operation_plan import snapshot

    plan = DesignPlan.model_validate(
        {
            "objective": "Copy the source profile",
            "parts": ["part"],
            "operation_plan": {
                "part_ids": ["part"],
                "steps": [
                    {
                        "id": "construct",
                        "part_id": "part",
                        "expected_effect": "Construct the complete measured profile",
                        "feature_ids": [f["id"] for f in target["proposal_document"]["features"]],
                        "requirement_ids": ["r1"],
                    }
                ],
            },
            "verification_bindings": [
                {"subject_id": "plate", "part_id": "part", "frame": "definition_local"}
            ],
        }
    )
    if not photo.store.get(photo.sid, "owner").get("design_plan"):
        from forge_mind.workspace.edit_service import read_edit_basis

        photo.store.set_design_plan_for_job(
            photo.sid,
            "owner",
            photo.job["id"],
            plan,
            expected_plan_version=0,
            contract=photo.contract,
            expected_current_revision=None,
            expected_edit_basis=read_edit_basis(photo.service, photo.sid, "owner"),
        )
    captured = snapshot(plan, photo.store.get(photo.sid, "owner")["plan_version"], photo.contract)
    directory = photo.store.directory(photo.sid, rid)
    directory.mkdir(parents=True)
    (directory / "part.step").write_bytes(data)
    document = {**target["proposal_document"], "production": {"role": "make"}}
    photo.store.add_revision(
        photo.sid,
        {
            "id": rid,
            "document": document,
            "code": "retained fake worker program",
            "contract": photo.contract.model_dump(mode="json"),
            "measurement": {
                "revision_id": rid,
                "geometry_valid": True,
                "artifact_hashes": {"part.step": hashlib.sha256(data).hexdigest()},
                "engine_fingerprint": "e" * 64,
                "contract_hash": photo.contract.fingerprint(),
            },
            "operation_record": {"plan_hash": captured["plan_hash"], "plan_snapshot": captured},
        },
    )
    photo.store.update(
        photo.sid,
        selected_revision=rid,
        plan_hash=captured["plan_hash"],
        design_plan=plan.model_dump(mode="json"),
        plan_contract_hash=photo.contract.fingerprint(),
        plan_version=1,
    )


def test_wrong_saved_candidate_correction_keeps_exact_source_target(photo):
    before = deepcopy(photo.store.session_record(photo.sid, "owner"))
    target = freeze(photo)
    observation = profiles.read_profile(photo.service, photo.sid, "owner", target["observation_id"])
    assert observation["result"]["calibration"]["independent_check_status"] == "not_run"
    assert observation["result"]["status"] == "advisory"
    assert len(observation["result"]["components"][0]["loops"]) == 3
    assert photo.store.session_record(photo.sid, "owner")["contract"] == before["contract"]
    revision(photo, target, "b" * 32, b"wrong fake STEP")
    assert (
        generation.photo_closure(photo.service, photo.sid, "owner", "b" * 32)["status"]
        == "comparison_needed"
    )
    failed = generation.compare_candidate(
        photo.service,
        photo.sid,
        "owner",
        photo.job["id"],
        "b" * 32,
        timeout=30,
        cancelled=lambda: False,
    )
    assert failed["status"] == "needs_correction"
    revision(photo, target, "c" * 32, b"correct fake STEP")
    assert (
        generation.photo_closure(photo.service, photo.sid, "owner", "c" * 32)["status"]
        == "comparison_needed"
    )
    checked = generation.compare_candidate(
        photo.service,
        photo.sid,
        "owner",
        photo.job["id"],
        "c" * 32,
        timeout=30,
        cancelled=lambda: False,
    )
    assert checked["status"] == "checked" and checked["target_sha256"] == target["target_sha256"]
    assert (
        generation.read_photo_target(photo.service, photo.sid, "owner", "c" * 32, "plate")[
            "observation_sha256"
        ]
        == target["observation_sha256"]
    )
    assert (
        profiles.read_profile(photo.service, photo.sid, "owner", target["observation_id"])
        == observation
    )
    with pytest.raises(WorkspaceError):
        generation.read_photo_target(photo.service, photo.sid, "other", "c" * 32, "plate")
    (photo.store.directory(photo.sid, "c" * 32) / "part.step").write_bytes(b"tampered")
    assert (
        generation.photo_closure(photo.service, photo.sid, "owner", "c" * 32)["status"]
        == "needs_correction"
    )


def test_blocked_grounding_requires_no_fabricated_facts_or_corners(photo):
    generation.observe_inventory(
        photo.service, photo.sid, "owner", photo.job["id"], timeout=30, cancelled=lambda: False
    )
    record = generation._read(photo.service, photo.sid, "owner")
    value = generation.PhotoGrounding(
        inventory_observation_id=record["inventory_ids"][0],
        blocked_reason="No source scale is available",
    )
    result = generation.ground_target(
        photo.service,
        photo.sid,
        "owner",
        photo.job["id"],
        value,
        photo.contract,
        timeout=30,
        cancelled=lambda: False,
    )
    assert (
        result["stage"] == "unsupported"
        and generation._read(photo.service, photo.sid, "owner")["target"] is None
    )


def test_frozen_target_cannot_change_during_correction_or_publish_after_cancel(photo):
    target = freeze(photo)
    record = generation._read(photo.service, photo.sid, "owner")
    record["target"] = {**target, "part_id": "other"}
    with pytest.raises(WorkspaceError, match="cannot be replaced"):
        generation._write(photo.service, photo.sid, "owner", photo.job["id"], record)
    with photo.store.connect() as con:
        con.execute("UPDATE jobs SET cancel=1 WHERE id=?", (photo.job["id"],))
    with pytest.raises(WorkspaceError, match="active job"):
        generation._write(
            photo.service,
            photo.sid,
            "owner",
            photo.job["id"],
            generation._read(photo.service, photo.sid, "owner"),
        )


def test_same_failed_comparison_is_not_retried_without_candidate_change(photo):
    target = freeze(photo)
    revision(photo, target, "d" * 32, b"wrong fake STEP")
    generation.record_failure(
        photo.service,
        photo.sid,
        "owner",
        photo.job["id"],
        "compare",
        "d" * 32,
        "Unsupported wire curve",
    )
    context = generation.generation_context(
        photo.service, photo.sid, "owner", "d" * 32, photo.contract
    )
    assert context["stage"] == "correct" and "Unsupported wire" in context["closure"]["reason"]


def test_inventory_failure_consumes_one_action_and_no_provider_retry(photo, monkeypatch):
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor

    def fail(*args, **kwargs):
        raise ValueError("deterministic image bound failure")

    monkeypatch.setattr(profiles, "process_revision", fail)
    service = WorkspaceService(
        photo.store, Executor("trusted_local"), lambda **kwargs: SimpleNamespace()
    )
    state = service._run_agent_job({**photo.job, "max_actions": 16, "max_seconds": 60})
    assert state == "needs_review"
    events = photo.store.get(photo.sid, "owner")["events"]
    assert len([e for e in events if e["kind"] == "photo_generation"]) == 1
    assert (
        "deterministic image bound failure"
        in generation._read(service, photo.sid, "owner")["blocked_reason"]
    )


def test_thin_open_rectangular_band_cannot_become_filled_rectangle():
    import numpy as np

    # Independent C-region witness: center is exterior; near-complete boundary
    # lies close to all four rectangle edges, with a narrow opening at right.
    points = np.array(
        [
            [10, 5.05],
            [10, 10],
            [0, 10],
            [0, 0],
            [10, 0],
            [10, 4.95],
            [9.95, 4.95],
            [9.95, 0.05],
            [0.05, 0.05],
            [0.05, 9.95],
            [9.95, 9.95],
            [9.95, 5.05],
            [10, 5.05],
        ]
    )
    with pytest.raises(ValueError, match="material region"):
        _rectangle_fit(points, (10, 10), np.diag([0.01, 0.01, 1]))


@pytest.mark.parametrize("extra", ["duplicate", "other_made", "reference"])
def test_one_copy_scope_counts_every_installed_occurrence_without_native(extra):
    target = SimpleNamespace(id="plate", production=SimpleNamespace(role="make"))
    parts = [target]
    roots = [SimpleNamespace(definition="plate")]
    if extra == "duplicate":
        roots.append(SimpleNamespace(definition="plate"))
    else:
        parts.append(
            SimpleNamespace(
                id="other",
                production=SimpleNamespace(role="make" if extra == "other_made" else "reference"),
            )
        )
        roots.append(SimpleNamespace(definition="other"))
    document = SimpleNamespace(assembly=SimpleNamespace(parts=parts, roots=roots, subassemblies=[]))
    with pytest.raises(WorkspaceError, match=r"one-copy|One-copy"):
        generation._one_copy_scope(document, "plate")


def test_visible_face_agreement_does_not_close_hidden_material_error(photo):
    target = freeze(photo)
    revision(photo, target, "f" * 32, b"hidden pocket fake STEP")
    closure = generation.compare_candidate(
        photo.service,
        photo.sid,
        "owner",
        photo.job["id"],
        "f" * 32,
        timeout=30,
        cancelled=lambda: False,
    )
    assert closure["measurement"]["result"]["symmetric_residual_interval_mm"] == [0, 0.02]
    assert closure["whole_body"]["status"] == "fail"
    assert closure["status"] == "needs_correction"


def test_context_only_role_does_not_force_photo_copy_target(photo, monkeypatch):
    from forge_mind.workspace import photo_nominal

    # Role authentication/positive-context grammar has separate source-admission tests.
    monkeypatch.setattr(photo_nominal, "photo_reference_role", lambda session: "context")
    generation.observe_inventory(
        photo.service, photo.sid, "owner", photo.job["id"], timeout=30, cancelled=lambda: False
    )
    assert (
        generation.generation_context(photo.service, photo.sid, "owner", None, photo.contract)
        is None
    )
    generation.require_grounded(photo.service, photo.sid, "owner")
    assert (
        generation.photo_closure(photo.service, photo.sid, "owner", None)["status"]
        == "not_applicable"
    )


def test_deadline_is_checked_inside_publication_transaction(photo):
    generation.observe_inventory(
        photo.service, photo.sid, "owner", photo.job["id"], timeout=30, cancelled=lambda: False
    )
    before = generation._read(photo.service, photo.sid, "owner")
    changed = {**before, "blocked_reason": "late write"}
    with pytest.raises(WorkspaceError, match="deadline"):
        generation._write(photo.service, photo.sid, "owner", photo.job["id"], changed, deadline=0)
    assert generation._read(photo.service, photo.sid, "owner") == before


@pytest.mark.parametrize("repair", [True, False])
def test_scripted_author_cannot_skip_automatic_compare_or_exhaustion(photo, monkeypatch, repair):
    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace import (
        agent,
        author_images,
        completion,
        completion_inline,
        customer_construction,
        document_service,
    )
    from forge_mind.workspace.author_images import AuthorImages
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Action

    target = freeze(photo)
    revision(photo, target, "b" * 32, b"wrong fake STEP")
    calls = []

    class Client:
        def generate_structured_with_image(self, **kwargs):
            calls.append(json.loads(kwargs["user"]))
            action = (
                Action(
                    kind="document",
                    explanation="Apply located correction",
                    document_json=json.dumps(target["proposal_document"]),
                )
                if repair and len(calls) == 1
                else Action(
                    kind="finish", explanation="Attempt to finish without choosing any photo tool"
                )
            )
            return kwargs["output_model"].model_validate(action.model_dump()), LLMUsage(
                "scripted", 0, 0
            )

    service = agent.WorkspaceService(
        photo.store, Executor("trusted_local"), lambda **kwargs: Client()
    )
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "0")
    monkeypatch.setattr(
        service, "_load_revision", lambda sid, rid, **kwargs: photo.store.revision(sid, rid)
    )
    monkeypatch.setattr(
        author_images,
        "prepare_author_images",
        lambda *args, **kwargs: AuthorImages((), (), None, "fake_worker", "not_requested"),
    )
    monkeypatch.setattr(completion, "completion_context", lambda *args: {"premature_finish": False})
    monkeypatch.setattr(
        completion_inline, "author_checkpoint", lambda *args: {"status": "unverified"}
    )
    monkeypatch.setattr(customer_construction, "stage_construction", lambda *args, **kwargs: None)

    def fake_build(*args, **kwargs):
        revision(photo, target, "c" * 32, b"correct fake STEP")
        return photo.store.revision(photo.sid, "c" * 32)

    monkeypatch.setattr(document_service, "build_document", fake_build)
    state = service._run_agent_job(
        {
            **photo.job,
            "max_actions": 4,
            "max_seconds": 60,
            "preview_only": True,
            "delivery_requested": False,
        }
    )
    closure = generation.photo_closure(
        service, photo.sid, "owner", "c" * 32 if repair else "b" * 32
    )
    assert calls and all("photo_generation" in call for call in calls), (
        state,
        photo.store.get(photo.sid, "owner")["events"],
    )
    assert closure["target_sha256"] == target["target_sha256"]
    assert closure["status"] == ("checked" if repair else "needs_correction"), (
        closure,
        state,
        photo.store.get(photo.sid, "owner")["events"],
    )
    assert photo.calls.count("photo_profile_compare") == (2 if repair else 1)
    assert state == "needs_review"  # fake observer supplies no nominal/physical authority
    assert len(calls) == (2 if repair else 3)
