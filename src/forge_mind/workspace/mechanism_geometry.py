"""Bind a planar graph to actual native solids and independently check gear layout."""

from __future__ import annotations

import math
from typing import Any

import build123d as b

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.mechanism import pose_matrix, resolve_mechanism, study_mechanism
from forge_mind.workspace.topology import same_exported_solid, signature


def matrix_location(matrix: Any) -> Any:
    """Convert a shared rigid pose matrix without reconstructing Euler angles."""
    return b.Plane(
        origin=tuple(matrix[:3, 3]),
        x_dir=tuple(matrix[:3, 0]),
        z_dir=tuple(matrix[:3, 2]),
    ).location


def place_components(
    document: DesignDocument, shapes: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert document.mechanism is not None
    graph = resolve_mechanism(document.mechanism, document.values())
    study = study_mechanism(graph)
    if study["current"]["status"] != "solved":
        raise ValueError(
            f"Mechanism current pose is {study['current']['status']}: residual {study['current']['residual_mm_equivalent']:.6g}, remaining local DOF {study['current']['driven_local_dof']}"
        )
    bindings = []
    placed = dict(shapes)
    for component in graph["components"]:
        fid = component["feature"]
        solids = list(shapes[fid].solids())
        if len(solids) != 1 and not document.assembly:
            raise ValueError(f"Component {component['id']} must own exactly one solid")
        location = matrix_location(
            pose_matrix(study["current"]["poses"][component["id"]], graph["frame"])
        )
        if document.assembly:
            from forge_mind.workspace.assembly import named_solids

            children = []
            if shapes[fid].label.startswith("ck_occ_"):
                children = [shapes[fid]]
            else:
                children = list(named_solids(shapes[fid]).values())
                # Solid enumeration drops Python labels; retain compiler leaf names.
                for label, child in zip(named_solids(shapes[fid]), children, strict=True):
                    child.label = label
        else:
            children = solids
        moved = []
        for child in children:
            solid = location * child
            solid.label = child.label if document.assembly else "ck_component_" + component["id"]
            moved.append(solid)
            bindings.append(
                {
                    "component": component["id"],
                    "feature": fid,
                    "name": component["name"],
                    "signature": signature(solid),
                    "label": solid.label,
                }
            )
        if document.assembly:
            by_label = {s.label: s for s in moved}

            def reconstruct(node: Any, by_label: dict[str, Any] = by_label) -> Any:
                if node.label in by_label:
                    return by_label[node.label]
                return b.Compound(
                    children=[reconstruct(c) for c in node.children], label=node.label
                )

            placed[fid] = reconstruct(shapes[fid])
        else:
            placed[fid] = moved[0]
    study["gear_layout"] = gear_layout(document, graph, study)
    study["graph"] = graph
    return placed, {
        "id": "__mechanism__",
        "state": "computed",
        "mechanism": study,
        "component_bindings": bindings,
    }


def gear_layout(
    document: DesignDocument, graph: dict[str, Any], study: dict[str, Any]
) -> list[dict[str, Any]]:
    features = {f.id: f for f in document.features}
    components = {c["id"]: c for c in graph["components"]}
    checks = []
    for gear in graph["gears"]:
        if document.assembly:
            checks.append(
                {
                    "id": gear["id"],
                    "status": "unverified",
                    "reason": "Assembly gear geometry needs a definition-level gear interface",
                }
            )
            continue
        a, other = (features[components[gear[k]]["feature"]] for k in ("a", "b"))
        check: dict[str, Any] = {
            "id": gear["id"],
            "status": "unverified",
            "scope": "External spur gear parameters and center spacing only; no tooth-contact or strength claim",
        }
        if a.kind != "gear" or other.kind != "gear":
            check["reason"] = "Both coupled outputs must be native gear features"
        else:
            p, q = (
                {**f.params, **{k: document.values()[v] for k, v in f.bindings.items()}}
                for f in (a, other)
            )
            expected = p["module"] * (p["teeth"] + q["teeth"]) / 2
            distance_errors = [
                abs(math.dist(s["poses"][gear["a"]][:2], s["poses"][gear["b"]][:2]) - expected)
                for s in [study["current"], *study["samples"]]
            ]
            matched = (
                abs(p["module"] - q["module"]) < 1e-9
                and abs(p.get("pressure_angle", 20) - q.get("pressure_angle", 20)) < 1e-9
                and abs(gear["ratio"] - q["teeth"] / p["teeth"]) < 1e-9
            )
            z_overlap = min(
                components[gear["a"]]["z"] + p["thickness"] / 2,
                components[gear["b"]]["z"] + q["thickness"] / 2,
            ) - max(
                components[gear["a"]]["z"] - p["thickness"] / 2,
                components[gear["b"]]["z"] - q["thickness"] / 2,
            )
            check.update(
                status="pass"
                if matched and max(distance_errors) <= 1e-5 and z_overlap > 0
                else "fail",
                expected_center_distance_mm=expected,
                max_sampled_center_error_mm=max(distance_errors),
                face_overlap_mm=z_overlap,
                speed_ratio=-p["teeth"] / q["teeth"],
                method="r=m*z/2; external center distance r1+r2; omega2/omega1=-z1/z2",
            )
        checks.append(check)
    return checks


def bind_exported(
    record: dict[str, Any],
    topology: dict[str, Any],
    step_hash: str,
    document_hash: str,
    imported: Any,
) -> dict[str, Any]:
    bindings = []
    used = set()
    solids = list(imported.solids())

    def descendants(node: Any) -> list[Any]:
        return [child for c in node.children for child in [c, *descendants(c)]]

    children = descendants(imported)
    for item in record["component_bindings"]:
        named = [
            child
            for child in children
            if child.label == item.get("label", "ck_component_" + item["component"])
        ]
        if len(named) != 1 or len(named[0].solids()) != 1:
            raise ValueError(
                f"Exported component {item['component']} lost its unique assembly name"
            )
        matches = [
            i for i, solid in enumerate(solids) if same_exported_solid(solid, named[0].solids()[0])
        ]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError(
                f"Exported component {item['component']} is missing or ambiguous; cannot bind motion to CAD"
            )
        index = matches[0]
        entity = next(
            e for e in topology["entities"] if e["kind"] == "body" and e["body_index"] == index
        )
        used.add(index)
        bindings.append(
            {
                **item,
                "source_signature": item["signature"],
                "signature": entity["signature"],
                "body_index": index,
            }
        )
    if len(used) != len(solids):
        raise ValueError("Export contains unbound mechanism solids")
    return {
        **record["mechanism"],
        "bindings": bindings,
        "step_sha256": step_hash,
        "document_sha256": document_hash,
        "binding_method": "Trusted compiler component names preserved in STEP assembly; exact imported topological identity joins named children to measured bodies",
    }
