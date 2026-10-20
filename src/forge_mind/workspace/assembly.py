"""Reusable assembly definitions, bounded expansion and exported interface evidence."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import build123d as b
import numpy as np

from forge_mind.validation_provenance import file_sha256
from forge_mind.validator.geometry import extract_cylinders
from forge_mind.workspace.capacity import enforce_limit, optional_limit
from forge_mind.workspace.document import DesignDocument, Occurrence, digest
from forge_mind.workspace.topology import same_exported_solid, signature


def occurrence_label(path: str) -> str:
    return "ck_occ_" + "_".join(str(len(part)) + "_" + part for part in path.split("/"))


def scalar(value: float | str, values: dict[str, float]) -> float:
    result = values[value] if isinstance(value, str) else value
    if not math.isfinite(result):
        raise ValueError("Assembly value must be finite")
    return result


def location(instance: Occurrence, values: dict[str, float]) -> Any:
    rx, ry, rz = (scalar(v, values) for v in instance.rotation)
    return b.Location(
        tuple(scalar(v, values) for v in instance.position),
        (rx, ry, rz),
    )


def expand(document: DesignDocument) -> list[dict[str, Any]]:
    spec = document.assembly
    assert spec
    definitions = {p.id: p for p in spec.parts}
    groups = {s.id: s for s in spec.subassemblies}
    result: list[dict[str, Any]] = []

    values = document.values()
    limit = optional_limit("OCCURRENCE_LIMIT")
    pending: list[Any] = [(iter(spec.roots), "", b.Location(), frozenset())]
    checked: set[str] = set()
    while pending:
        iterator, parent, transform, ancestors = pending[-1]
        item = next(iterator, None)
        if item is None:
            pending.pop()
            continue
        path = parent + "/" + item.id if parent else item.id
        if path in checked:
            raise ValueError("Sibling occurrence IDs must be unique")
        checked.add(path)
        loc = transform * location(item, values)
        if item.definition in definitions:
            result.append({"path": path, "definition": item.definition, "location": loc})
            enforce_limit(len(result), limit, "physical occurrences")
        elif item.definition in groups and item.definition not in ancestors:
            pending.append(
                (iter(groups[item.definition].children), path, loc, ancestors | {item.definition})
            )
        else:
            raise ValueError("Missing or cyclic assembly definition: " + item.definition)
    return result


def validate_assembly(document: DesignDocument) -> None:
    spec = document.assembly
    assert spec
    ids = [p.id for p in spec.parts] + [s.id for s in spec.subassemblies]
    if len(set(ids)) != len(ids):
        raise ValueError("Part and subassembly definition IDs must be unique")
    for part in spec.parts:
        DesignDocument(
            variables=document.variables,
            relations=document.relations,
            features=part.features,
            outputs=[part.output],
        )
        if len({p.name for p in part.ports}) != len(part.ports):
            raise ValueError("Part port names must be unique")
        for port in part.ports:
            if (
                sum(v * v for v in port.axis) < 1e-12
                or scalar(port.diameter, document.values()) <= 0
            ):
                raise ValueError("Port needs a nonzero axis and positive diameter")
            for value in port.origin:
                scalar(value, document.values())
    rows = expand(document)
    labels = [occurrence_label(row["path"]) for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("Occurrence paths collide in STEP names; rename ambiguous IDs")
    parts = {p.id: p for p in spec.parts}
    ports = {row["path"] + "." + p.name for row in rows for p in parts[row["definition"]].ports}
    if len({m.id for m in spec.mates}) != len(spec.mates):
        raise ValueError("Mate IDs must be unique")
    for mate in spec.mates:
        if mate.a == mate.b or {mate.a, mate.b} - ports or mate.min_clearance > mate.max_clearance:
            raise ValueError("Mate needs two existing ports and ordered clearance bounds")
    if document.mechanism:
        values = document.values()
        nonzero_roots = sorted(
            r.id
            for r in spec.roots
            if any(scalar(v, values) != 0 for v in (*r.position, *r.rotation))
        )
        if nonzero_roots:
            raise ValueError(
                "Mechanism controls assembly root placements; keep root transforms zero; "
                f"nonzero root IDs: {nonzero_roots}"
            )


def compile_assembly(
    document: DesignDocument,
    assets: dict[str, Path],
    directory: Path | None = None,
    cache_source: Path | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    from forge_mind.workspace.kernel import compile_document
    from forge_mind.workspace.measure import engine_fingerprint
    from forge_mind.workspace.processing import copy_artifact, regular_file

    spec = document.assembly
    assert spec
    rows = expand(document)
    used = {r["definition"] for r in rows}
    shapes: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    previous: dict[str, Any] = {}
    if cache_source and regular_file(cache_source / "parts.json"):
        previous = json.loads((cache_source / "parts.json").read_text(encoding="utf-8"))
    engine = engine_fingerprint()
    values = document.values()
    for part in spec.parts:
        if part.id not in used:
            continue
        cacheable = all(feature.kind != "python" for feature in part.features)
        bindings = {v for f in part.features for v in f.bindings.values()}
        bindings.update(
            v for f in part.features for point in f.points for v in point if isinstance(v, str)
        )
        key = digest(
            {
                "features": [f.model_dump() for f in part.features],
                "output": part.output,
                "values": {v: values[v] for v in sorted(bindings)},
                "assets": {
                    f.asset_id: file_sha256(assets[f.asset_id]) for f in part.features if f.asset_id
                },
                "engine": engine,
                # Opaque source can have undeclared dependencies. Distinct definitions
                # must not share captured bytes even when their source is identical.
                "uncached_definition": None if cacheable else part.id,
            }
        )
        filename = "part_" + key[:32] + ".brep"
        if any(v["file"] == filename and v["key"] != key for v in manifest.values()):
            raise ValueError("Compiled part cache filename collision")
        cached = previous.get(part.id, {})
        reused = False
        if cacheable and cache_source and cached.get("key") == key:
            source = cache_source / filename
            if regular_file(source) and file_sha256(source) == cached.get("sha256"):
                from OCP.BRep import BRep_Builder
                from OCP.BRepTools import BRepTools
                from OCP.TopoDS import TopoDS_Shape

                native = TopoDS_Shape()
                BRepTools.Read_s(native, BytesIO(source.read_bytes()), BRep_Builder())
                if native.IsNull():
                    raise ValueError("Could not deserialize compiled part")
                shape = b.Compound.cast(native)
                reused = True
        if not reused:
            shape, _ = compile_document(
                DesignDocument(
                    variables=document.variables,
                    relations=document.relations,
                    features=part.features,
                    outputs=[part.output],
                ),
                assets,
            )
        if not shape.is_valid or len(shape.solids()) != 1 or shape.volume <= 0:
            raise ValueError(f"Part {part.id} must build exactly one valid solid")
        shapes[part.id] = shape.solids()[0]
        if directory:
            target = directory / filename
            materialized = next((v for v in manifest.values() if v["file"] == filename), None)
            if materialized:
                # Different definitions may intentionally share identical geometry.
                # Reuse only bytes this build already materialized and verified;
                # arbitrary preexisting destinations still reach exclusive copy.
                if not regular_file(target) or file_sha256(target) != materialized["sha256"]:
                    raise ValueError("Shared compiled part cache changed during assembly build")
                if reused and cached["sha256"] != materialized["sha256"]:
                    raise ValueError("Shared compiled part cache content mismatch")
            elif reused and cache_source:
                copy_artifact(cache_source / filename, target)
            else:
                # Native filenames can exceed Windows MAX_PATH despite relative
                # worker paths. Python handles the path; OCCT only sees bytes.
                buffer = BytesIO()
                if not b.export_brep(shape, buffer):
                    raise ValueError("Could not serialize compiled part")
                target.write_bytes(buffer.getvalue())
            target_hash = file_sha256(target)
            step = "definition_" + part.id + ".step"
            b.export_step(shape, directory / step)
            manifest[part.id] = {
                "key": key,
                "sha256": target_hash,
                "file": filename,
                "step": step,
            }
        entry = {"id": part.id, "state": "reused" if reused else "built", "key": key}
        if not cacheable:
            entry.update(
                signature=signature(shapes[part.id]),
                cache_policy="disabled_for_python_source",
                history_scope="Retained editable source; deterministic regeneration is unverified",
            )
        history.append(entry)
    roots: dict[str, Any] = {}
    occurrence_bindings = []
    rows_by_root: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_root.setdefault(row["path"].split("/", 1)[0], []).append(row)
    for root in spec.roots:
        children: dict[str, Any] = {}
        for row in rows_by_root.get(root.id, []):
            # build123d.moved shares the source TShape; its STEP writer names that
            # definition rather than the occurrence. Copy before placement so each
            # exported occurrence keeps its unique identity across the roundtrip.
            solid = deepcopy(shapes[row["definition"]]).moved(row["location"])
            solid.label = occurrence_label(row["path"])
            children[row["path"]] = solid
            occurrence_bindings.append(
                {"path": row["path"], "definition": row["definition"], "label": solid.label}
            )

        child_paths: dict[str, dict[str, None]] = {}
        for path in children:
            segments = path.split("/")
            for index in range(1, len(segments)):
                parent = "/".join(segments[:index])
                child = "/".join(segments[: index + 1])
                child_paths.setdefault(parent, {})[child] = None
        nodes = dict(children)
        for parent in sorted(child_paths, key=lambda path: path.count("/"), reverse=True):
            nodes[parent] = b.Compound(
                children=[nodes[child] for child in child_paths[parent]],
                label="ck_group_" + parent.replace("/", "__"),
            )
        roots[root.id] = nodes[root.id]
    if document.mechanism:
        from forge_mind.workspace.mechanism_geometry import place_components

        roots, motion = place_components(document, roots)
        history.append(motion)
    if directory:
        (directory / "parts.json").write_text(json.dumps(manifest), encoding="utf-8")
    history.append(
        {
            "id": "__assembly__",
            "bindings": occurrence_bindings,
            "parts": manifest,
            "built": sum(h.get("state") == "built" for h in history),
            "reused": sum(h.get("state") == "reused" for h in history),
        }
    )
    return b.Compound(children=list(roots.values())), history


def named_solids(shape: Any) -> dict[str, Any]:
    result = {}

    pending = [shape]
    while pending:
        node = pending.pop()
        if node.label.startswith("ck_occ_"):
            if node.label in result or len(node.solids()) != 1:
                raise ValueError("Ambiguous assembly occurrence label")
            result[node.label] = node.solids()[0]
        else:
            pending.extend(reversed(node.children))
    return result


def assembly_evidence(
    document: DesignDocument, shape: Any, record: dict[str, Any]
) -> dict[str, Any]:
    """Join exported named occurrences to measured bodies and verify cylindrical ports."""
    spec = document.assembly
    assert spec
    exported = named_solids(shape)
    solids = list(shape.solids())
    rows = expand(document)
    definitions = {p.id: p for p in spec.parts}
    values = document.values()
    poses = None
    if document.mechanism:
        from forge_mind.workspace.mechanism import pose_matrix, resolve_mechanism, study_mechanism
        from forge_mind.workspace.mechanism_geometry import matrix_location

        graph = resolve_mechanism(document.mechanism, values)
        poses = study_mechanism(graph)["current"]["poses"]
        roots = {c["feature"]: c["id"] for c in graph["components"]}
    occurrences, ports = [], {}
    used: set[int] = set()
    for row in rows:
        label = occurrence_label(row["path"])
        solid = exported.get(label)
        if solid is None:
            raise ValueError("Export lost occurrence " + row["path"])
        matches = [i for i, s in enumerate(solids) if same_exported_solid(s, solid)]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError("Export occurrence identity is ambiguous")
        used.add(matches[0])
        part = definitions[row["definition"]]
        loc = row["location"]
        if poses:
            pose = poses[roots[row["path"].split("/")[0]]]
            loc = matrix_location(pose_matrix(pose, graph["frame"])) * loc
        cylinders = extract_cylinders(solid)
        for port in part.ports:
            local = np.array([scalar(v, values) for v in port.origin])
            origin = np.array(tuple(b.Vertex(*local).moved(loc).center()))
            end = np.array(tuple(b.Vertex(*(local + np.array(port.axis))).moved(loc).center()))
            axis = end - origin
            axis /= np.linalg.norm(axis)
            diameter = scalar(port.diameter, values)
            found = []
            for c in cylinders:
                ca, cp = np.array(c.axis_dir), np.array(c.axis_point)
                if (
                    c.concave == (port.kind == "bore")
                    and abs(2 * c.radius - diameter) < 1e-5
                    and np.linalg.norm(np.cross(axis, ca)) < 1e-6
                    and np.linalg.norm(np.cross(origin - cp, ca)) < 1e-5
                    and c.axial_extent is not None
                    and c.axial_extent[0] - 1e-5
                    <= float(np.dot(origin, ca))
                    <= c.axial_extent[1] + 1e-5
                ):
                    found.append(c)
            ports[row["path"] + "." + port.name] = {
                "status": "pass" if found else "fail",
                "kind": port.kind,
                "diameter_mm": diameter,
                "origin": origin.tolist(),
                "axis": axis.tolist(),
                "method": "Exported cylindrical material side, radius, axis and axial span",
            }
        occurrences.append(
            {
                "path": row["path"],
                "definition": part.id,
                "name": part.name,
                "body_index": matches[0],
                "volume_mm3": float(solid.volume),
                "center_mm": list(solid.center(b.CenterOf.MASS)),
                "bbox_min_mm": list(solid.bounding_box().min),
                "bbox_max_mm": list(solid.bounding_box().max),
                "cylindrical_surfaces": [
                    {
                        "surface_id": f"cylinder_{index}",
                        "radius_mm": c.radius,
                        "diameter_mm": 2 * c.radius,
                        "axis_point_mm": list(c.axis_point),
                        "axis_direction": list(c.axis_dir),
                        "concave": c.concave,
                        "axial_extent_mm": list(c.axial_extent)
                        if c.axial_extent is not None
                        else None,
                    }
                    for index, c in enumerate(cylinders)
                ],
            }
        )
    if len(used) != len(solids):
        raise ValueError("Unbound exported assembly bodies")
    mates = []
    for mate in spec.mates:
        a, other = ports[mate.a], ports[mate.b]
        axis = np.array(a["axis"])
        delta = np.array(other["origin"]) - np.array(a["origin"])
        radial = float(np.linalg.norm(np.cross(delta, axis)))
        axial = abs(float(np.dot(delta, axis)))
        aligned = float(np.linalg.norm(np.cross(axis, other["axis"]))) < 1e-6
        clearance = (
            a["diameter_mm"] - other["diameter_mm"]
            if a["kind"] == "bore"
            else other["diameter_mm"] - a["diameter_mm"]
        )
        ok = (
            a["status"] == other["status"] == "pass"
            and a["kind"] != other["kind"]
            and radial < 1e-5
            and axial <= mate.axial_tolerance + 1e-5
            and aligned
            and mate.min_clearance - 1e-5 <= clearance <= mate.max_clearance + 1e-5
        )
        mates.append(
            {
                "id": mate.id,
                "a": mate.a,
                "b": mate.b,
                "status": "pass" if ok else "fail",
                "clearance_mm": clearance,
                "radial_error_mm": radial,
                "axial_offset_mm": axial,
                "scope": "Nominal exported cylindrical interface only; tolerance, retention and capacity unverified",
            }
        )
    return {
        "occurrences": occurrences,
        "ports": ports,
        "mates": mates,
        "unique_parts": len({r["definition"] for r in rows}),
        "occurrence_count": len(rows),
        "build": {k: record[k] for k in ("built", "reused")},
        "bom": [
            {
                "id": p.id,
                "name": p.name,
                "quantity": sum(r["definition"] == p.id for r in rows),
                "source": p.source,
            }
            for p in spec.parts
            if any(r["definition"] == p.id for r in rows)
        ],
        "status": "fail"
        if any(p["status"] == "fail" for p in ports.values())
        or any(m["status"] == "fail" for m in mates)
        else "unverified",
        "scope": "Occurrence identity, part quantities and declared nominal interfaces measured; whole assembly function unverified",
    }


def preserved_assembly(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    old, new = baseline.get("assembly"), candidate.get("assembly")
    if not old:
        return True, "No prior assembly"
    if not new:
        return False, "Assembly definitions were removed"
    before, after = (
        DesignDocument.model_validate(baseline),
        DesignDocument.model_validate(candidate),
    )
    identities = {r["path"]: r["definition"] for r in expand(after)}
    missing = [r["path"] for r in expand(before) if identities.get(r["path"]) != r["definition"]]
    for part in old["parts"]:
        replacement = next((p for p in new["parts"] if p["id"] == part["id"]), None)
        if replacement is None or {p["name"] for p in part["ports"]} - {
            p["name"] for p in replacement["ports"]
        }:
            missing.append(part["id"] + ".ports")
    new_mates = {m["id"]: m for m in new["mates"]}
    missing.extend(m["id"] for m in old["mates"] if new_mates.get(m["id"]) != m)
    return not missing, "Removed assembly identities: " + ", ".join(
        missing
    ) if missing else "Occurrence definitions and interface identities retained"
