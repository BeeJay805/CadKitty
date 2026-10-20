from copy import deepcopy

from forge_mind.workspace.completion_work import remaining_work


def row():
    return {
        "id": "fit",
        "requirement_ids": ["physical_fit"],
        "dependencies": [],
        "status": "unverified",
        "checks": [],
        "geometry_declared": False,
        "missing_part_ids": [],
        "missing_occurrence_paths": [],
    }


def obligations():
    return [
        {
            "requirement_id": "physical_fit",
            "declared": True,
            "physical_validation": ["Print and test the supplied fixture"],
        }
    ]


def test_external_only_work_stays_unverified_without_repeated_construction():
    work = row()
    before = deepcopy(work)
    result = remaining_work([work], obligations())
    assert result["next_target"] is None
    assert result["external_obligations"][0]["subsystem_id"] == "fit"
    assert work == before and work["status"] == "unverified"


def test_physical_obligation_cannot_excuse_missing_requested_parts_or_failed_checks():
    work = row()
    work.update(status="missing", missing_part_ids=["mount"])
    assert remaining_work([work], obligations())["next_work_category"] == "construction"
    work.update(status="checked_fail", missing_part_ids=[], checks=[{"status": "fail"}])
    assert remaining_work([work], obligations())["next_work_category"] == "repair"


def test_prepared_checks_and_undeclared_review_remain_actionable():
    work = row()
    work["checks"] = [{"status": "unverified", "reason": "Not run"}]
    assert remaining_work([work], obligations())["next_work_category"] == "check"
    work["checks"] = []
    assert remaining_work([work], [])["next_work_category"] == "review"


def test_repair_precedes_construction_and_respects_dependency_geometry():
    other = {**row(), "id": "cover", "status": "missing", "missing_part_ids": ["cover"]}
    work = {**row(), "status": "checked_fail", "checks": [{"status": "fail"}]}
    assert remaining_work([other, work], obligations())["next_target"] == "fit"
    work["dependencies"] = ["cover"]
    assert remaining_work([other, work], obligations())["next_target"] == "cover"


def test_one_external_declaration_cannot_hide_an_undeclared_requirement():
    work = row()
    work["requirement_ids"].append("unexplained_function")
    assert remaining_work([work], obligations())["next_target"] == "fit"


def test_passing_cad_checks_do_not_hide_external_obligations():
    work = row()
    work.update(status="checked_pass", geometry_declared=True, checks=[{"status": "pass"}])
    result = remaining_work([work], obligations())
    assert result["next_target"] is None
    assert result["external_obligations"][0]["subsystem_id"] == "fit"


def test_core_operation_plan_still_requires_its_missing_construction():
    steps = [
        {"step_id": "cut_bore", "kind": "construction", "graph_present": False},
        {"step_id": "inspect", "kind": "inspection", "graph_present": False},
    ]
    result = remaining_work([], [], steps)
    assert result["next_target"] == "cut_bore"
    steps[0]["graph_present"] = True
    assert remaining_work([], [], steps)["next_target"] is None
