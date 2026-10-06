"""The protocol itself: clutter levels and the clutter-level classifier (no Isaac Sim dependency)."""

from .clutter_levels import (
    CLUTTER_CONFIGS,
    COMPLEXITY_LEVELS,
    NEIGHBOR_RADIUS,
    apply_complexity_correction,
    count_neighbors,
)
