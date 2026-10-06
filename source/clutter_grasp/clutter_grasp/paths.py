"""Repository paths.

    source/clutter_grasp/clutter_grasp/assets/   hand model + benchmark object USDs (in git)
    weights/                                     classifier, PPO policies, GG-CNN, SAM (downloaded, not in git)
    data/                                        generated datasets and EGAD meshes (not in git)
    results/                                     benchmark outputs (not in git)

Scripts parse arguments before Isaac Sim starts and cannot import this package at that
point, so they define the same paths relative to their own file.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
ASSETS_DIR = PACKAGE_DIR / "assets"
OBJECTS_DIR = ASSETS_DIR / "objects"
WEIGHTS_DIR = REPO_ROOT / "weights"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
