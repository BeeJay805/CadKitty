"""The deterministic standard-hardware catalog -- Forge-Mind's McMaster-style brain.

Public API: select a verified part by id, search/list, generate its geometry, and validate
hardware fits. The LLM never invents standard mechanical geometry -- it picks a catalog id and this
package generates the part deterministically. Importing the package registers every family's
builder; the registry materialises lazily on first lookup (bd_warehouse stays lazy).
"""

from __future__ import annotations

# Importing each family module registers its builder with the registry.
from forge_mind.agentic.catalog import bearings as bearings
from forge_mind.agentic.catalog import fasteners as fasteners
from forge_mind.agentic.catalog import gears as gears
from forge_mind.agentic.catalog import nuts as nuts
from forge_mind.agentic.catalog import washers as washers
from forge_mind.agentic.catalog.registry import (
    CatalogError,
    CatalogIssue,
    CatalogPart,
    catalog_counts,
    catalog_geometry,
    get_catalog_part,
    list_catalog_family,
    search_catalog,
    suggest_catalog_ids,
)
from forge_mind.agentic.catalog.stacks import (
    HardwareSpec,
    bearing_mount,
    catalog_bearing,
    catalog_gear,
    catalog_nut,
    catalog_screw,
    catalog_washer,
    gear_pair,
    gear_pair_geometry,
    planetary_set,
    resolve_part,
    screw_counterbore,
    screw_stack,
)

__all__ = [
    "CatalogError",
    "CatalogIssue",
    "CatalogPart",
    "HardwareSpec",
    "bearing_mount",
    "catalog_bearing",
    "catalog_counts",
    "catalog_gear",
    "catalog_geometry",
    "catalog_nut",
    "catalog_screw",
    "catalog_washer",
    "gear_pair",
    "gear_pair_geometry",
    "get_catalog_part",
    "list_catalog_family",
    "planetary_set",
    "resolve_part",
    "screw_counterbore",
    "screw_stack",
    "search_catalog",
    "suggest_catalog_ids",
]
