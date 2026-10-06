"""Clutter levels of the protocol and the neighbour-count rule used to confirm them.

A scene is described by the clutter around ONE target object:

    isolated   the target alone on the table
    C0_easy    2-4 neighbours, 10-20 cm apart
    C1_medium  4-6 neighbours,  8-12 cm apart
    C2_hard    8-10 neighbours,  3-8 cm apart

Scenes are generated with the spawn rules below, then confirmed by the clutter classifier
(see ``classifier.py``). The classifier's prediction is corrected with the number of
neighbours within ``NEIGHBOR_RADIUS`` of the target (``apply_complexity_correction``).
"""

import numpy as np

CLUTTER_CONFIGS = {
    "C0_easy": {
        "num_neighbors": (2, 4),  # 2-3 neighbouring objects
        "min_clearance": 0.10,    # 10 cm minimum spacing
        "max_clearance": 0.20,    # 20 cm maximum spacing
        "description": "Easy - Few isolated neighbors",
    },
    "C1_medium": {
        "num_neighbors": (4, 6),  # 4-5 neighbouring objects
        "min_clearance": 0.08,    # 8 cm minimum spacing
        "max_clearance": 0.12,    # 12 cm maximum spacing
        "description": "Medium - Moderate clutter",
    },
    "C2_hard": {
        "num_neighbors": (8, 10),  # 8-10 neighbouring objects
        "min_clearance": 0.03,     # 3 cm minimum spacing
        "max_clearance": 0.08,     # 8 cm maximum spacing
        "description": "Hard - Dense clutter",
    },
}

# Classifier output index for each clutter level.
COMPLEXITY_LEVELS = {"C0_easy": 0, "C1_medium": 1, "C2_hard": 2}

# Two objects are neighbours when their normalised image-plane positions are closer than this.
NEIGHBOR_RADIUS = 0.07


def count_neighbors(obj_idx: int, spatial_features_list: list, radius: float = NEIGHBOR_RADIUS) -> int:
    """Count objects whose (obj_x, obj_y) spatial features lie within ``radius`` of object ``obj_idx``.

    ``spatial_features_list[i]`` is the 6-feature vector of object i (see
    ``classifier.SPATIAL_FEATURES``) or None when the object is not visible.
    """
    if spatial_features_list[obj_idx] is None:
        return 0

    obj_x = spatial_features_list[obj_idx][0]
    obj_y = spatial_features_list[obj_idx][1]

    neighbor_count = 0
    for other_idx, other_features in enumerate(spatial_features_list):
        if other_idx == obj_idx or other_features is None:
            continue
        distance = np.sqrt((obj_x - other_features[0]) ** 2 + (obj_y - other_features[1]) ** 2)
        if distance < radius:
            neighbor_count += 1

    return neighbor_count


def apply_complexity_correction(predicted: int, neighbor_count: int, obj_name: str = "target") -> int:
    """Correct the classifier's clutter level using the neighbour count.

    0-1 neighbours -> C0, 2 neighbours -> C1, 3+ neighbours -> C2, but only where the
    classifier's prediction disagrees with that count (rules below are exactly those
    used to produce the paper's results).
    """
    corrected = predicted
    correction_reason = None

    if predicted == 0 and neighbor_count > 1:
        if neighbor_count <= 2:
            corrected = 1
            correction_reason = f"C0 has {neighbor_count} neighbors (>1) → C1"
        else:
            corrected = 2
            correction_reason = f"C0 has {neighbor_count} neighbors (>2) → C2"

    elif predicted == 1:
        if neighbor_count <= 1:
            corrected = 0
            correction_reason = f"C1 has {neighbor_count} neighbor(s) (≤1) → C0"
        elif neighbor_count > 2:
            corrected = 2
            correction_reason = f"C1 has {neighbor_count} neighbors (>2) → C2"

    elif predicted == 2 and neighbor_count <= 2:
        if neighbor_count <= 1:
            corrected = 0
            correction_reason = f"C2 has {neighbor_count} neighbor(s) (≤1) → C0"
        else:
            corrected = 1
            correction_reason = f"C2 has {neighbor_count} neighbors (≤2) → C1"

    if correction_reason:
        print(f"[CORRECTION] {obj_name}: {correction_reason}")

    return corrected
