"""Native-free content identity for the installed workspace geometry engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from forge_mind.validation_provenance import validation_fingerprint


def engine_fingerprint() -> str:
    digest = hashlib.sha256(validation_fingerprint().encode())
    for package in (
        "build123d",
        "cadquery-ocp",
        "numpy",
        "Pillow",
        "scikit-image",
        "scipy",
        "trimesh",
        "scikit-fem",
        "pytetwild",
        "gmsh",
    ):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        digest.update(f"{package}={version}".encode())
    root = Path(__file__).parent.parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
