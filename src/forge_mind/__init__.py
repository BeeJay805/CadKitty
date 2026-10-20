"""Forge-Mind: AI-native CAD pipeline.

intent -> blueprint -> PartDSL -> compiler -> validator -> revision.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("forge-mind")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"
