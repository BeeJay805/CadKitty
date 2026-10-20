"""Display colors through complete same-invocation exported body correspondence.

The caller authenticates the document and obtains assembly_evidence from the
same imported STEP whose solids it renders. This projection never guesses a
body identity and never changes geometry, production intent or acceptance.
"""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forge_mind.workspace.preview_selection import select_preview_bodies

ROLE_COLORS = {
    "make": "#D68B3C",
    "buy": "#65A877",
    "reference": "#4774B0",
    "unspecified": "#929292",
}
ROLE_LABELS = {
    "make": "make",
    "buy": "buy",
    "reference": "reference only",
    "unspecified": "role undeclared",
}
MAX_LEGEND_ENTRIES = 8
MAX_LABEL_CHARACTERS = 24


@dataclass(frozen=True)
class PreviewAppearance:
    body_colors: tuple[str, ...]
    legend: tuple[tuple[str, str], ...]
    omitted_labels: int
    diagnostic: str


def preview_appearance(
    document: Mapping[str, object] | None,
    verified_assembly: Mapping[str, object] | None,
    body_count: int,
) -> PreviewAppearance:
    """Return exact body-index colors, or an entirely neutral full-view fallback."""
    selection = select_preview_bodies(document, verified_assembly, body_count)

    def fallback(reason: str) -> PreviewAppearance:
        return PreviewAppearance((), (), 0, "Part roles unavailable: " + reason)

    if selection.status == "full_context_fallback":
        return fallback(selection.diagnostic)
    assert isinstance(document, Mapping)
    spec = document["assembly"]
    assert isinstance(spec, Mapping)
    parts = spec["parts"]
    assert isinstance(parts, list)
    definitions: dict[str, tuple[str, str, str]] = {}
    for part in parts:
        assert isinstance(part, Mapping)
        identifier = part["id"]
        assert isinstance(identifier, str)
        production = part.get("production")
        if production is None:
            production = {}
        if not isinstance(production, Mapping):
            return fallback("Malformed production declaration.")
        role = production.get("role", "unspecified")
        if not isinstance(role, str) or role not in ROLE_COLORS:
            return fallback("Unknown production role.")
        appearance = production.get("appearance")
        if appearance is None:
            appearance = {}
        if not isinstance(appearance, Mapping):
            return fallback("Malformed color declaration.")
        color = appearance.get("color")
        if color is None:
            color = ROLE_COLORS[role]
        if not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
            return fallback("Malformed color declaration.")
        name = part.get("name", identifier)
        if not isinstance(name, str) or not name.strip():
            return fallback("Missing part name.")
        name = " ".join(name.split())
        if len(name) > MAX_LABEL_CHARACTERS:
            name = name[: MAX_LABEL_CHARACTERS - 1] + "…"
        definitions[identifier] = (name, role, color)
    assert isinstance(verified_assembly, Mapping)
    rows = verified_assembly["occurrences"]
    assert isinstance(rows, list)
    colors = [""] * body_count
    counts: Counter[str] = Counter()
    for row in rows:
        assert isinstance(row, Mapping)
        identifier, index = row["definition"], row["body_index"]
        assert isinstance(identifier, str) and isinstance(index, int)
        colors[index] = definitions[identifier][2]
        counts[identifier] += 1
    labels = [
        (color, f"{name}\n{ROLE_LABELS[role]}, qty {counts[identifier]}")
        for identifier, (name, role, color) in definitions.items()
        if counts[identifier]
    ]
    return PreviewAppearance(
        tuple(colors),
        tuple(labels[:MAX_LEGEND_ENTRIES]),
        max(0, len(labels) - MAX_LEGEND_ENTRIES),
        "Colors show declared appearance or role defaults; not material or physical approval.",
    )


def render_primary_preview(
    directory: Path,
    shape: Any,
    document: Mapping[str, object] | None,
    verified_assembly: Mapping[str, object] | None,
) -> tuple[Any, dict[str, Any]]:
    """Render once through same-shape body colors, with a visible neutral fallback.

    Call only after authoritative exports. Cleaning cached triangulation changes
    display tessellation, not STEP geometry; measurement reopens saved files.
    """
    import numpy as np
    import trimesh
    from build123d import export_stl, import_step
    from OCP.BRepTools import BRepTools

    from forge_mind.viz.render import render_workspace_preview

    solids = list(shape.solids())
    appearance = preview_appearance(document, verified_assembly, len(solids))
    facts = asdict(appearance)
    image = directory / "render.png"
    display_mesh = None
    palette = None
    if appearance.body_colors:
        try:
            BRepTools.Clean_s(shape.wrapped)
            meshes, colors = [], []
            for index, solid in enumerate(solids):
                intermediate = directory / f"preview-body-{index}.stl"
                try:
                    if not export_stl(
                        solid, str(intermediate), tolerance=0.15, angular_tolerance=0.35
                    ):
                        raise ValueError("Display body tessellation export failed")
                    mesh = trimesh.load(str(intermediate), force="mesh")
                    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                        raise ValueError("Display body tessellation is empty")
                    meshes.append(mesh)
                    color = appearance.body_colors[index]
                    rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
                    colors.append(np.tile(rgb, (len(mesh.faces), 1)))
                finally:
                    intermediate.unlink(missing_ok=True)
            display_mesh = trimesh.util.concatenate(meshes)
            palette = np.concatenate(colors)
        except (TimeoutError, MemoryError):
            raise
        except Exception as exc:
            facts["diagnostic"] = (
                "Part roles unavailable: display tessellation failed (" + type(exc).__name__ + ")."
            )
            facts["body_colors"], facts["legend"], facts["omitted_labels"] = (), (), 0
            display_mesh = None
    note = facts["diagnostic"]
    if facts["omitted_labels"]:
        note = f"{facts['omitted_labels']} part labels omitted. " + note
    neutral = directory / "preview.stl"
    neutral_palette = np.array(
        [[int(ROLE_COLORS["unspecified"][i : i + 2], 16) / 255 for i in (1, 3, 5)]]
    )
    try:
        if display_mesh is None:
            preview_shape = import_step(str(directory / "part.step"))
            if not export_stl(preview_shape, str(neutral), tolerance=0.15, angular_tolerance=0.35):
                raise ValueError("Complete display tessellation export failed")
            render_workspace_preview(
                neutral, image, face_colors=neutral_palette, appearance_note=note
            )
            bounds = trimesh.load(str(neutral), force="mesh").bounds.copy()
        else:
            try:
                render_workspace_preview(
                    neutral,
                    image,
                    display_mesh=display_mesh,
                    face_colors=palette,
                    legend=appearance.legend,
                    appearance_note=note,
                )
                bounds = display_mesh.bounds.copy()
            except (TimeoutError, MemoryError):
                raise
            except Exception as exc:
                facts["diagnostic"] = (
                    "Part roles unavailable: color rendering failed (" + type(exc).__name__ + ")."
                )
                facts["body_colors"], facts["legend"], facts["omitted_labels"] = (), (), 0
                preview_shape = import_step(str(directory / "part.step"))
                if not export_stl(
                    preview_shape, str(neutral), tolerance=0.15, angular_tolerance=0.35
                ):
                    raise ValueError("Complete display tessellation export failed") from exc
                render_workspace_preview(
                    neutral, image, face_colors=neutral_palette, appearance_note=facts["diagnostic"]
                )
                bounds = trimesh.load(str(neutral), force="mesh").bounds.copy()
    finally:
        neutral.unlink(missing_ok=True)
    facts["scope"] = (
        "Display-only colors and declared part roles; complete opaque geometry retained."
    )
    return bounds, facts
