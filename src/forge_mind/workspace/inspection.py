"""Direct dependency evidence and revision-pinned pages for large editable documents."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import defaultdict
from typing import Any

from forge_mind.workspace.action_payloads import InputValidationError, read_design_arguments
from forge_mind.workspace.document import DesignDocument, digest


def author_revision_context(revision: dict[str, Any]) -> dict[str, Any]:
    """Bound automatic context, while immutable documents remain complete and queryable."""
    document = revision.get("document")
    if not document:
        return revision
    result = {k: v for k, v in revision.items() if k not in {"code", "diff"}}
    assembly = document.get("assembly")
    parts = assembly.get("parts", []) if assembly else []
    features = (
        [f for p in parts for f in p.get("features", [])]
        if assembly
        else document.get("features", [])
    )
    large = len(json.dumps(document)) > 24000 or any(
        len(f.get("code") or "") > 4096 for f in features
    )
    result["document_hash"] = digest(document)
    result["document_is_summary"] = bool(assembly) or large
    if assembly:
        relations = document.get("relations", [])
        result["binding_editing"] = {
            "action": "assembly_edit",
            "schema_query": "assembly_edit",
            "input_count": len(document.get("variables", {})),
            "derived_count": len(relations),
            "derived_preview": [
                {
                    "name": relation["name"],
                    "depends_on": sorted(
                        {
                            value
                            for value in (relation.get("a"), relation.get("b"))
                            if isinstance(value, str)
                        }
                    ),
                }
                for relation in relations[:12]
            ],
            "usage": (
                "parameters edits existing input variables only. Omitted relations survive "
                "assembly_edit. Use promote_relations={existing_relation_name: new_value} "
                "to make a derived binding an input. Use derive_variables=[VariableRelation] "
                "to make an existing input derived. Do not repeat transition names in "
                "ordinary variables or relations. Both transitions validate atomically. "
                "read_design variables|relations retrieves complete revision-pinned bindings."
            ),
            "scope": "Declared dependencies and edit syntax, not measured geometry or fit.",
        }
    if not large:
        if assembly:
            result["document"] = {
                **document,
                "assembly": {
                    **assembly,
                    "parts": [
                        {k: v for k, v in p.items() if k != "features"}
                        | {"feature_count": len(p.get("features", []))}
                        for p in parts
                    ],
                },
            }
            result["omitted_document_fields"] = ["assembly.parts[].features"]
        return result
    result.pop("parameters", None)
    result.pop("editable_parameters", None)
    result["parameter_count"] = len(document.get("variables", {}))

    def feature_summary(feature: dict[str, Any]) -> dict[str, Any]:
        source = feature.get("code") or ""
        return {
            "id": feature["id"],
            "name": feature.get("name"),
            "kind": feature["kind"],
            "input_count": len(feature.get("inputs", [])),
            "source_characters": len(source),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        }

    counts = {
        "variables": len(document.get("variables", {})),
        "relations": len(document.get("relations", [])),
        "features": len(features),
        "parts": len(parts),
        "roots": len(assembly.get("roots", [])) if assembly else 0,
        "subassemblies": len(assembly.get("subassemblies", [])) if assembly else 0,
        "mates": len(assembly.get("mates", [])) if assembly else 0,
    }
    result["document"] = {
        "representation": "summary_only_not_a_design_document",
        "name": document.get("name"),
        "document_hash": result["document_hash"],
        "counts": counts,
        "preview": {
            "variables": [
                {"name": k, "value": v} for k, v in list(document.get("variables", {}).items())[:12]
            ],
            "features": [
                feature_summary(f) | {"part_id": p["id"]}
                for p in parts[:12]
                for f in p.get("features", [])[:1]
            ][:12]
            if assembly
            else [feature_summary(f) for f in features[:12]],
            "parts": [
                {"id": p["id"], "name": p.get("name"), "feature_count": len(p.get("features", []))}
                for p in parts[:12]
            ],
        },
        "read_pages": {
            "action": "read_design",
            "revision_id": revision["id"],
            "collections": [
                "variables",
                "relations",
                "parts",
                "features",
                "occurrences",
                "metadata",
                "roots",
                "subassemblies",
                "mates",
            ],
            "feature_lookup": "query=features, document_json={scope: part_id for assemblies, feature_id: stable feature ID, limit: 1}",
            "continuation": "Follow next_cursor to complete the requested collection; pages bind this revision and full document hash.",
        },
        "editing": "Use parameters or read_part/edit_part/assembly_edit to retain omitted content. Never submit this summary as a replacement document. Read all relevant pages and full part definitions before a full replacement.",
    }
    # Per-feature history and large body inventories are available through explicit
    # inspection; they must not defeat the document context budget on every turn.
    measurement = result.get("measurement", {})
    facts = measurement.get("facts", {})
    compact_facts = {}
    for key, value in facts.items():
        if len(json.dumps(value)) <= 4000:
            compact_facts[key] = value
        else:
            compact_facts[key] = {
                "is_summary": True,
                "item_count": len(value) if isinstance(value, (list, dict)) else None,
                "sha256": digest(value),
                "retrieve": "Complete data remains in this revision's evidence.json; inspect the revision for measured details",
            }
    result["measurement"] = {**measurement, "facts": compact_facts}
    return result


def design_graph(document: DesignDocument) -> dict[str, Any]:
    values = document.values()
    nodes: dict[str, dict[str, str]] = {}
    edges: set[tuple[str, str]] = set()

    def node(kind: str, name: str) -> str:
        key = f"{kind}/{name}"
        nodes[key] = {"id": key, "kind": kind, "name": name}
        return key

    def bind(target: str, refs: Any) -> None:
        for ref in refs:
            if isinstance(ref, str):
                edges.add((node("variable", ref), target))

    for name in values:
        node("variable", name)
    relations = []
    for relation in document.relations:
        relation_refs = [v for v in (relation.a, relation.b) if isinstance(v, str)]
        bind(node("variable", relation.name), relation_refs)
        relations.append(
            {
                **relation.model_dump(),
                "value": values[relation.name],
                "inputs": relation_refs,
                "depends_on": sorted(set(relation_refs)),
            }
        )

    def features(items: Any, part: str | None = None) -> None:
        for feature in items:
            key = node(
                "part_feature" if part else "feature",
                f"{part}/{feature.id}" if part else feature.id,
            )
            bind(key, feature.bindings.values())
            bind(key, [v for point in feature.points for v in point])
            for parent in feature.inputs:
                edges.add(
                    (
                        node(
                            "part_feature" if part else "feature",
                            f"{part}/{parent}" if part else parent,
                        ),
                        key,
                    )
                )
            if part:
                edges.add((key, node("part", part)))

    features(document.features)
    definitions = []
    if document.assembly:
        for part in document.assembly.parts:
            features(part.features, part.id)
            refs = {v for f in part.features for v in f.bindings.values()}
            refs.update(
                v for f in part.features for point in f.points for v in point if isinstance(v, str)
            )
            ports = [v for port in part.ports for v in (*port.origin, port.diameter)]
            bind(node("part", part.id), ports)
            refs.update(v for v in ports if isinstance(v, str))
            definitions.append(
                {
                    "id": part.id,
                    "name": part.name,
                    "feature_count": len(part.features),
                    "bindings": sorted(refs),
                    "depends_on": sorted(refs),
                    "ports": [p.name for p in part.ports],
                }
            )
        placements = [("root", r) for r in document.assembly.roots]
        placements.extend((g.id, c) for g in document.assembly.subassemblies for c in g.children)
        for placement_group, occurrence in placements:
            bind(
                node("placement", f"{placement_group}/{occurrence.id}"),
                (*occurrence.position, *occurrence.rotation),
            )
    if document.motion_rig:
        for group in document.motion_rig.groups:
            bind(node("motion", group.id), group.pivot)
        for driver in document.motion_rig.drivers:
            bind(node("motion", driver.id), [driver.reference])
    if document.mechanism:
        mechanism = document.mechanism
        for component in mechanism.components:
            bind(
                node("mechanism", f"component/{component.id}"),
                [component.x, component.y, component.z, component.angle],
            )
        for joint in mechanism.joints:
            bind(
                node("mechanism", f"joint/{joint.id}"),
                [*joint.pa, *joint.pb, joint.angle, joint.axis],
            )
        for gear in mechanism.gears:
            bind(node("mechanism", f"gear/{gear.id}"), [gear.ratio, gear.phase])
        bind(node("mechanism", "driver"), [mechanism.driver.angle])
    graph = {
        "document_hash": digest(document.model_dump()),
        "graph_semantics": "direct_v2",
        "variables": document.variables,
        "relations": relations,
        "parts": definitions,
        "nodes": list(nodes.values()),
        "edges": [{"source": a, "target": b} for a, b in sorted(edges)],
        "mates": [m.model_dump() for m in document.assembly.mates] if document.assembly else [],
        "coverage": "Direct explicit CAD bindings. Query reachability for transitive impact. Loads, manufacturing inputs and implicit physical relationships are outside this graph.",
        "scope": "Declared dependency graph, not solved constraints or physical validation.",
    }
    return graph


def reachable(graph: dict[str, Any], node_ids: list[str], *, reverse: bool = False) -> list[str]:
    known = {node["id"] for node in graph["nodes"]}
    if set(node_ids) - known:
        raise ValueError("Dependency query references an unknown node")
    adjacent: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        adjacent[b if reverse else a].append(a if reverse else b)
    seen = set(node_ids)
    pending = list(node_ids)
    while pending:
        for target in adjacent[pending.pop()]:
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return sorted(seen - set(node_ids))


def variable_impact(graph: dict[str, Any], names: list[str]) -> dict[str, list[str]]:
    kinds = {
        "part": "parts",
        "placement": "placements",
        "motion": "motion",
        "feature": "features",
        "mechanism": "mechanism",
    }
    result: dict[str, list[str]] = {kind: [] for kind in kinds.values()}
    for key in reachable(graph, ["variable/" + name for name in names]):
        kind, name = key.split("/", 1)
        if kind in kinds:
            result[kinds[kind]].append(name)
    return result


def query_document(
    document: DesignDocument,
    revision_id: str,
    collection: str,
    *,
    scope: str | None = None,
    cursor: str | None = None,
    limit: int = 100,
    feature_id: str | None = None,
) -> dict[str, Any]:
    """A continuation binds revision, complete document hash, collection and scope."""
    arguments = read_design_arguments(
        collection, {"scope": scope, "cursor": cursor, "limit": limit, "feature_id": feature_id}
    )
    scope, cursor, limit, feature_id = (
        arguments.scope,
        arguments.cursor,
        arguments.limit,
        arguments.feature_id,
    )
    identity = {
        "revision_id": revision_id,
        "document_hash": document.fingerprint(),
        "collection": collection,
        "scope": scope,
        "feature_id": feature_id,
    }
    offset = 0
    if cursor is not None:
        try:
            decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            if not isinstance(decoded, dict):
                raise ValueError("Malformed cursor")
            offset = decoded.pop("offset")
            if decoded != identity or type(offset) is not int or offset < 0:
                raise ValueError("Cursor does not belong to this revision and query")
        except (ValueError, KeyError, UnicodeError) as exc:
            raise InputValidationError(
                "invalid_cursor",
                "read_design.cursor",
                "Cursor does not belong to this revision and query; restart without a cursor",
            ) from exc
    rows: list[Any]
    if feature_id is not None and collection != "features":
        raise ValueError("Feature ID applies only to a features query")
    if collection == "variables":
        rows = [
            {"name": key, "value": value, "editable": key in document.variables}
            for key, value in document.values().items()
        ]
    elif collection == "relations":
        rows = list(document.relations)
    elif collection == "parts":
        rows = (
            [
                {"id": p.id, "name": p.name, "feature_count": len(p.features), "output": p.output}
                for p in document.assembly.parts
            ]
            if document.assembly
            else []
        )
    elif collection == "features":
        items = document.features
        if document.assembly:
            selected = next((p for p in document.assembly.parts if p.id == scope), None)
            if selected is None:
                raise ValueError("Assembly feature query requires an existing part scope")
            items = selected.features
        rows = list(items)
        if feature_id is not None:
            rows = [f for f in items if f.id == feature_id]
            if not rows:
                raise ValueError("Unknown feature ID in this scope")
    elif collection in {"roots", "subassemblies", "mates"}:
        rows = list(getattr(document.assembly, collection)) if document.assembly else []
    elif collection == "metadata":
        rows = [
            {"name": k, "value": v}
            for k, v in document.model_dump().items()
            if k not in {"variables", "relations", "features", "assembly"}
        ]
    elif collection == "occurrences":
        from forge_mind.workspace.assembly import expand

        rows = (
            [{"path": row["path"], "definition": row["definition"]} for row in expand(document)]
            if document.assembly
            else []
        )
        if scope:
            rows = [
                row for row in rows if row["path"] == scope or row["path"].startswith(scope + "/")
            ]
    elif collection in {"nodes", "edges"}:
        rows = design_graph(document)[collection]
    else:
        raise ValueError("Unknown document collection")
    if offset > len(rows):
        raise ValueError("Cursor offset exceeds this collection")
    end = min(offset + limit, len(rows))
    next_cursor = None
    if end < len(rows):
        next_cursor = base64.urlsafe_b64encode(
            json.dumps({**identity, "offset": end}, sort_keys=True).encode()
        ).decode()
    return {
        **identity,
        "items": [
            row.model_dump() if hasattr(row, "model_dump") else row for row in rows[offset:end]
        ],
        "total": len(rows),
        "offset": offset,
        "returned": end - offset,
        "has_more": next_cursor is not None,
        "next_cursor": next_cursor,
        "coverage": "complete_collection_pagination",
    }
