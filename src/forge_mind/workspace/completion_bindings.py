"""Shared descriptions of existing completion bindings; no evidence or verdicts."""

from types import MappingProxyType

PREPARED_CHECK_TOOLS = MappingProxyType(
    {
        ("fit", "nominal_fit"): frozenset(
            {"hole_pattern", "mount_grid", "planar_face_pair", "projected_pair", "axial_cavity"}
        ),
        ("clearance", "static_clearance"): frozenset({"clearance_volume"}),
    }
)

MATE_BINDING_DESCRIPTION = (
    "connection/nominal_fit uses FunctionalInterface.mate_ids, not duty.check_ids. "
    "Current passing measured cylindrical mates must connect at least two participants "
    "in one graph covering every participant, with no outside endpoint. Empty state_ids "
    "are valid; selected states must be reference states. Retention and capacity remain unverified."
)

STATE_BINDING_DESCRIPTION = (
    "Prepared nominal/static checks and nominal connections allow empty state_ids; "
    "selected states must be reference states. travel/sampled_clearance requires nonempty "
    "motion-state selection. Each entire driver_values dictionary must equal a verified "
    "requested configuration. Motion checks generate deterministic samples, not arbitrary "
    "requested poses; bind recorded configurations or known reference/end configurations."
)

PARTICIPANT_BINDING_DESCRIPTION = (
    "Prepared checks each select a nonempty subset of these participants; their union must "
    "cover every participant. Travel requires at least one interface participant to move, "
    "not every participant. Sampled travel also requires exported bindings for every "
    "participant; unassigned participants may remain fixed."
)

CHECK_BINDING_DESCRIPTION = (
    "; ".join(
        f"{kind}/{scope}: prepared {', '.join(sorted(tools))}"
        for (kind, scope), tools in PREPARED_CHECK_TOOLS.items()
    )
    + ". Each prepared request selects a nonempty participant subset; their union must equal "
    "all interface participants. Empty state_ids are valid; selected states must be reference "
    "states. Current service-verified exact-request receipts are required. "
    + MATE_BINDING_DESCRIPTION
    + " travel/sampled_clearance uses current verified motion evidence, not check_ids: at least "
    "one participant moves in a bound motion rig, every participant has an exported binding, "
    "and nonempty motion states exactly match requested configurations. Evidence must be "
    "current, complete, error-free and discrete_only. Other travel scopes still require a "
    "current moving rig/mechanism participant. Unsupported duty/scope pairs remain unknown. "
    "External declarations do not waive missing construction or create physical proof."
)
