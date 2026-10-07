"""Shared helpers for the clutter-grasp protocol analysis scripts.

This module has three parts:
  1. paper plot style and display names for clutter conditions/controllers,
  2. loading of per-run ``results.json`` files, and
  3. the trial-level metric definitions that all figures share.

Metric definitions (used by analyze.py)
-------------------------------------------------------------------------
* Cost of a trial, C(tau): by default the trial's picking time
  (``picking_time``, seconds; see ``trial_cost`` for the accepted key names
  and ``--cost`` to use any other numeric per-trial field such as energy).
  By default only successful trials count (time-to-success); a trial with no
  ``success`` key is treated as successful. ``--all_trials`` counts every trial.
* Trial outlier filter: for each (controller, object, condition) cell, a
  picking time is dropped when its robust z-score
  ``0.6745 * |t - median| / MAD`` exceeds 3.5 (MAD = median absolute
  deviation). Cells with fewer than 4 times, or with MAD == 0, are left as-is.
* D_{i,s}: the mean cost of object i in condition s, after the outlier
  filter (with the default cost: "time to successful grasp").
* Success rate SR_{i,s}: ``success_rate`` from the run's summary block
  (``overall_statistics`` or ``overall_stats``). If that is missing, it is the
  fraction of trials with ``success == True``.
"""
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Plot style (single-column paper figures)
# ---------------------------------------------------------------------------
FONT_SIZE_LABEL = 7
FONT_SIZE_TICK = 6
FONT_SIZE_LEGEND = 5
FONT_SIZE_ANNOT = 4
FONT_SIZE_CBAR = 6
TIGHT_PAD = 0.4


def apply_paper_style():
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # silence font-subsetting chatter
    warnings.filterwarnings("ignore", message="Tight layout not applied")  # narrow figures with few objects
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.family": "serif",
        "savefig.format": "pdf", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "cm",
        "axes.labelsize": FONT_SIZE_LABEL, "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK, "legend.fontsize": FONT_SIZE_LEGEND,
        "axes.titlesize": 8,
    })


# ---------------------------------------------------------------------------
# Clutter conditions. Run folders carry no suffix for the isolated baseline.
# ---------------------------------------------------------------------------
CLUTTER_LEVELS = ["C0_easy", "C1_medium", "C2_hard"]
CONDITIONS = ["isolated"] + CLUTTER_LEVELS

# Accepted spellings of each condition in results files (E_s notation of the paper, 0-3, names).
_COND_ALIASES = {
    "isolated": ["isolated", "baseline", "e0", "0", "none", ""],
    "C0_easy": ["c0_easy", "c0", "e1", "1", "easy"],
    "C1_medium": ["c1_medium", "c1", "e2", "2", "medium"],
    "C2_hard": ["c2_hard", "c2", "e3", "3", "hard"],
}
COND_ALIAS = {alias: cond for cond, aliases in _COND_ALIASES.items() for alias in aliases}


def normalize_condition(value) -> Optional[str]:
    """Map a condition written as 'C1_medium', 'E2', 2, 'medium', ... to its canonical name.

    None and '' mean the isolated baseline. Unknown values return None.
    """
    if value is None:
        return "isolated"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return COND_ALIAS.get(str(value).strip().lower())
COND_LABEL = {
    "isolated": r"$\epsilon_0$ (Baseline)",
    "C0_easy": r"$\mathcal{E}_1$ (Easy)",
    "C1_medium": r"$\mathcal{E}_2$ (Medium)",
    "C2_hard": r"$\mathcal{E}_3$ (Hard)",
}
COND_COLOR = {"isolated": "#3498db", "C0_easy": "#2ecc71",
              "C1_medium": "#f39c12", "C2_hard": "#e74c3c"}

# ---------------------------------------------------------------------------
# Controller display styles. Names not listed here get their own name as the
# label and a colour from FALLBACK_COLORS; override with --label / --color.
# ---------------------------------------------------------------------------
DEFAULT_STYLES = {
    "rl":              dict(label="RL (Isolated)", color="#e8c84a", marker="o", linestyle="-"),
    "heuristic":       dict(label="Heuristic", color="#ff7f0e", marker="s", linestyle=":"),
    "transformer":     dict(label="Distilled Policy", color="#2ca02c", marker="^", linestyle="--"),
    "rl_clutter":      dict(label="RL (Clutter env)", color="#9467bd", marker="D", linestyle="-."),
    "distilled":       dict(label="Distilled Clutter", color="#17becf", marker="P", linestyle=(0, (3, 1, 1, 1))),
    "real_experiment": dict(label="Real-World Experiment", color="#1f77b4", marker="o", linestyle="-"),
}
FALLBACK_COLORS = ["#d62728", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#1f77b4"]
FALLBACK_MARKERS = ["v", "<", ">", "X", "h", "*"]


def parse_key_values(items: Optional[List[str]], what: str) -> Dict[str, str]:
    """Turn ['a=x', 'b=y'] into {'a': 'x', 'b': 'y'} (order preserved)."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{what} must be NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_styles(names: List[str], labels: Dict[str, str], colors: Dict[str, str]) -> Dict[str, dict]:
    """Display style for every controller name: defaults, then CLI overrides."""
    styles, n_unknown = {}, 0
    for name in names:
        if name in DEFAULT_STYLES:
            st = dict(DEFAULT_STYLES[name])
        else:
            st = dict(label=name, color=FALLBACK_COLORS[n_unknown % len(FALLBACK_COLORS)],
                      marker=FALLBACK_MARKERS[n_unknown % len(FALLBACK_MARKERS)], linestyle="-")
            n_unknown += 1
        st["label"] = labels.get(name, st["label"])
        st["color"] = colors.get(name, st["color"])
        styles[name] = st
    return styles


def short_label(obj: str) -> str:
    """Tick label of an object: EGAD names (e.g. 'A24_0') print only their letter, others in full."""
    s = str(obj)
    if re.fullmatch(r"[A-Y]\d{2}_\d+", s):
        return s[0]
    return s if s else "?"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_run_dir(name: str, prefix: str = "single") -> Optional[Tuple[str, str]]:
    """'<prefix>_<object>[_<C0_easy|C1_medium|C2_hard>]_YYYYMMDD_HHMMSS' -> (object, condition)."""
    if not name.startswith(prefix + "_"):
        return None
    parts = name[len(prefix) + 1:].split("_")
    if len(parts) < 3 or not (parts[-1].isdigit() and len(parts[-1]) == 6) \
            or not (parts[-2].isdigit() and len(parts[-2]) == 8):
        return None
    core, cond = parts[:-2], "isolated"
    if len(core) >= 2 and "_".join(core[-2:]) in CLUTTER_LEVELS:
        core, cond = core[:-2], "_".join(core[-2:])
    obj = "_".join(core)
    return (obj, cond) if obj else None


def run_identity(data: dict, path: Path, prefix: str = "single") -> Optional[Tuple[str, str]]:
    """(object, condition) of one results file.

    Read from the JSON first: ``target_object`` (or ``object``) and ``condition``
    (or ``target_complexity``; missing/empty = isolated). Falls back to the run
    folder name '<prefix>_<object>[_<level>]_<date>_<time>' used by this repository.
    """
    obj = data.get("target_object", data.get("object"))
    if obj:
        raw = data["condition"] if "condition" in data else data.get("target_complexity")
        cond = normalize_condition(raw)
        if cond is None:
            print(f"[WARN] {path}: unknown condition {raw!r}, skipped")
            return None
        return str(obj), cond
    for folder in (path.parent.parent, path.parent):
        parsed = parse_run_dir(folder.name, prefix)
        if parsed:
            return parsed
    return None


def load_runs(base_dir, prefix: str = "single") -> Dict[str, Dict[str, dict]]:
    """Read every results file under base_dir into {object: {condition: results dict}}.

    Any ``*.json`` (searched recursively) that is a dict with a ``trial_results``
    list and an identifiable object/condition (see ``run_identity``) is a run;
    other JSON files are ignored. This repository writes
    ``<run folder>/results/results.json``; if such a file has no
    ``trial_results`` list, the sibling ``trial_results.json`` is used.
    Files are read in sorted path order, so if an (object, condition) pair
    appears twice, the later one wins.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise SystemExit(f"Controller directory not found: {base_dir}")
    runs: Dict[str, Dict[str, dict]] = {}
    for res_file in sorted(base_dir.rglob("*.json")):
        if res_file.name == "trial_results.json":
            continue  # companion of results.json, read below when needed
        try:
            data = json.loads(res_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if not isinstance(data.get("trial_results"), list):
            tr_file = res_file.parent / "trial_results.json"
            if res_file.name != "results.json" or not tr_file.exists():
                continue
            tr = json.loads(tr_file.read_text())
            if isinstance(tr, dict):
                tr = tr.get("trial_results", [])
            data["trial_results"] = tr if isinstance(tr, list) else []
        ident = run_identity(data, res_file, prefix)
        if ident is None:
            continue
        obj, cond = ident
        runs.setdefault(obj, {})[cond] = data
    if not runs:
        raise SystemExit(f"No results files (JSON with target_object, condition and trial_results) in {base_dir}")
    return runs


# Cost settings, set once by analyze.py from --cost / --all_trials.
COST_KEY: Optional[str] = None     # None = picking time (first of TIME_KEYS present)
ALL_TRIALS: bool = False           # False = successful trials only
TIME_KEYS = ("picking_time", "time_to_success", "time", "duration", "elapsed_time", "pick_time", "cost")


def trial_cost(trial: dict) -> Optional[float]:
    """Cost C(tau) of one trial: the --cost field, or else the first finite value among TIME_KEYS."""
    for k in ((COST_KEY,) if COST_KEY else TIME_KEYS):
        try:
            v = float(trial[k])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(v):
            return v
    return None


def trial_costs(trials: list) -> List[float]:
    """Costs of the trials that count: successful ones (a trial without 'success' counts as
    successful), or every trial with --all_trials."""
    out = []
    for t in trials or []:
        if not ALL_TRIALS and "success" in t and not t.get("success", False):
            continue
        v = trial_cost(t)
        if v is not None:
            out.append(v)
    return out


# Names used before the cost was made configurable.
trial_time = trial_cost
successful_times = trial_costs


def mad_filter(values, z: float = 3.5) -> np.ndarray:
    """Drop values whose robust z-score 0.6745*|x - median|/MAD exceeds z.

    Applied separately to each (controller, object, condition) cell. Skipped
    for cells with < 4 values or MAD == 0, or when z <= 0.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if z <= 0 or v.size < 4:
        return v
    dev = np.abs(v - np.median(v))
    mad = np.median(dev)
    if not np.isfinite(mad) or mad <= 0:
        return v
    return v[0.6745 * dev / mad <= z]


def success_rate(data: dict) -> float:
    """Success rate in [0, 1] of one run (summary block first, else from trials)."""
    stats = data.get("overall_statistics") or data.get("overall_stats") or {}
    p = stats.get("success_rate")
    if p is not None:
        return float(p)
    trials = data.get("trial_results") or []
    if not trials or "success" not in trials[0]:
        return float("nan")
    return float(np.mean([bool(t.get("success", False)) for t in trials]))


def trials_table(runs: Dict[str, Dict[str, dict]]) -> pd.DataFrame:
    """One row per counted trial: object, condition, time (= cost; no outlier filtering yet)."""
    rows = [(obj, cond, v) for obj, conds in runs.items() for cond, data in conds.items()
            for v in trial_costs(data.get("trial_results"))]
    return pd.DataFrame(rows, columns=["object", "condition", "time"])


def lighten(color, amount: float):
    """Blend a colour toward white (0 = unchanged, 1 = white)."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
