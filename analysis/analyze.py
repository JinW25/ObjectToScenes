#!/usr/bin/env python3
"""Clutter-grasp protocol analysis: one entry point for simulated and real-world results.

Simulated figures (need one or more --controller NAME=DIR):
  ranking_heatmap_<ref>.pdf/.csv     Per-object difficulty ranking of the reference
                                     controller (--reference) in every condition.
  absolute_gap_violin_all_ctrls.pdf  Picking-time gap A_{i,s}^pi of every controller
  absolute_gap_violin_all_ctrls.csv  relative to the reference's isolated baseline.
  absolute_gap_summary.csv           Mean/median gap per controller x condition.
  object_metrics.csv                 D_{i,s} and SR_{i,s} for every controller.

Real-world figures (need --csv clutter_compile.csv):
  time_trend_overlay.pdf/.csv  Time to successful grasp D_{i,s} per object and
                               condition: simulated --trend controllers as lines
                               (left axis), real-world picks as stars (right axis).
  coef_all_terms.pdf/.csv      Gamma GLM  log E[t] = a0 + A*clutter + C*difficulty
                               + D*clutter*difficulty, one fit per simulated
                               --controller (needs at least one --controller).

Real-world input (clutter_compile.csv), one row per object pick:
  obj_name, clutter_level (0 = isolated, 1/2/3 = easy/medium/hard clutter),
  object_execute_s (execution time summed over all attempts on that object
  until it was picked; the real-world protocol retries until success, so
  this is the real-world "time to successful grasp").

See common.py for the trial-level definitions (picking time, MAD outlier
filter, D_{i,s}, success rate) and README.md for example commands.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FuncFormatter
from scipy import stats

import common as C

REAL = "real_experiment"

# Axis labels; --cost_label replaces the cost name (defaults are the paper's labels).
LABEL_GAP = r"Picking Time Gap $A_{i,s}^{\pi}$ (s)"
LABEL_D = r"Time to Successful Grasp $D_{i,s}$"


def set_cost_label(cost_label: str):
    global LABEL_GAP, LABEL_D
    LABEL_GAP = cost_label + r" Gap $A_{i,s}^{\pi}$"
    LABEL_D = cost_label + r" $D_{i,s}$"

# ---------------------------------------------------------------------------
# Simulated results: per-object metrics
# ---------------------------------------------------------------------------
def object_metrics(runs: dict, outlier_z: float) -> pd.DataFrame:
    """One row per object with an isolated run: D_<cond>, D_std_<cond>, SR_<cond>, N_<cond>.

    D_<cond> is the mean picking time of successful trials after the MAD
    outlier filter (NaN if the condition was not run or had no success).
    """
    rows = []
    for obj in sorted(runs):
        if "isolated" not in runs[obj]:
            continue  # objects without a baseline run are not analysed
        row = {"object": obj}
        for cond in C.CONDITIONS:
            data = runs[obj].get(cond)
            times = C.mad_filter(C.trial_costs(data["trial_results"]), outlier_z) if data else np.array([])
            row[f"D_{cond}"] = float(np.nanmean(times)) if times.size else np.nan
            row[f"D_std_{cond}"] = float(np.nanstd(times)) if times.size >= 2 else np.nan
            row[f"N_{cond}"] = int(times.size)
            row[f"SR_{cond}"] = C.success_rate(data) if data else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("object")


def rank_1_is_best(values) -> np.ndarray:
    """Rank 1 = lowest picking time (easiest object). NaN stays NaN."""
    v = np.asarray(values, dtype=float)
    r = np.full_like(v, np.nan)
    finite = np.isfinite(v)
    order = np.argsort(v[finite])
    rr = np.empty(order.size)
    rr[order] = np.arange(1, order.size + 1)
    r[np.where(finite)[0]] = rr
    return r


def ranking_table(m: pd.DataFrame) -> pd.DataFrame:
    """Objects sorted by baseline time D_isolated; each condition ranked separately.

    The ranking shows whether the order of object difficulty survives clutter:
    an object ranked #1 in the baseline is the reference controller's fastest
    pick; if it moves to #10 under E3, clutter hurt it more than the others.
    """
    t = m[np.isfinite(m["D_isolated"].values)].sort_values("D_isolated")
    out = pd.DataFrame({"object": t.index})
    for cond in C.CONDITIONS:
        out[f"D_{cond}"] = t[f"D_{cond}"].values
        out[f"rank_{cond}"] = rank_1_is_best(t[f"D_{cond}"].values)
    return out


def gap_table(metrics: dict, ref: str, sr_threshold: float) -> pd.DataFrame:
    """Absolute picking-time gap A_{i,s}^pi = D_{i,s}^pi - D_{i,0}^ref (seconds).

    The baseline is the reference controller's own isolated time for the same
    object, so the reference at epsilon_0 is 0 by construction and every other
    (controller, condition) is "seconds slower than the reference's
    uncluttered pick". Every object with a finite gap is kept; an object is
    only *flagged* (low_sr=True) when the controller's success rate in that
    condition, or the reference's isolated success rate, is below sr_threshold.
    Objects are listed in order of their mean baseline time across controllers.
    """
    allobj = sorted(set().union(*(m.index for m in metrics.values())))
    base = pd.DataFrame({c: metrics[c]["D_isolated"].reindex(allobj) for c in metrics})
    mean_base = base.mean(axis=1, skipna=True).dropna()
    order = sorted(mean_base.items(), key=lambda kv: kv[1])  # stable sort, as in the original
    objs = [o for o, _ in order]
    ref_m = metrics[ref].reindex(objs)
    ref_iso = ref_m["D_isolated"].values
    ref_ok = np.isfinite(ref_m["SR_isolated"].values) & (ref_m["SR_isolated"].values >= sr_threshold)
    rows = []
    for ctrl, m in metrics.items():
        mm = m.reindex(objs)
        for cond in C.CONDITIONS:
            gap = mm[f"D_{cond}"].values - ref_iso
            sr = mm[f"SR_{cond}"].values
            ok = np.isfinite(sr) & (sr >= sr_threshold) & ref_ok
            for i in np.where(np.isfinite(gap))[0]:
                rows.append(dict(controller=ctrl, condition=cond, object=objs[i], gap_s=gap[i],
                                 D=mm[f"D_{cond}"].values[i], D_ref_isolated=ref_iso[i],
                                 success_rate=sr[i], low_sr=not ok[i]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulated figures
# ---------------------------------------------------------------------------
def plot_ranking_heatmap(rt: pd.DataFrame, path: Path):
    n = len(rt)
    rows = ["C2_hard", "C1_medium", "C0_easy", "isolated"]  # top -> bottom
    names = [r"$\mathcal{E}_3$ (Hard)", r"$\mathcal{E}_2$ (Medium)", r"$\mathcal{E}_1$ (Easy)",
             r"$\epsilon_0$ Baseline"]
    ranks = np.vstack([rt[f"rank_{c}"].values for c in rows])
    fig, ax = plt.subplots(figsize=(5.0, len(rows) * 0.38 + 0.9))
    im = ax.imshow(ranks, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=max(n, 1), interpolation="nearest")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(names, fontweight="bold", fontsize=C.FONT_SIZE_TICK)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rt["object"].tolist(), rotation=45, ha="right", fontsize=C.FONT_SIZE_TICK)
    ax.set_xlabel(r"Objects $O_i$", fontweight="bold", fontsize=C.FONT_SIZE_LABEL)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Ranking (1 = easiest, lower time)", rotation=270, labelpad=25,
                   fontweight="bold", fontsize=C.FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=C.FONT_SIZE_TICK)
    for i in range(ranks.shape[0]):
        for j in range(n):
            if np.isfinite(ranks[i, j]):
                ax.text(j, i, f"#{int(round(ranks[i, j]))}", ha="center", va="center", fontsize=4.0,
                        color="white" if ranks[i, j] >= 0.65 * n else "black", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout(pad=C.TIGHT_PAD)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def draw_violin_box(ax, pos, data, color, width, names):
    """Violin (distribution over objects) + box (quartiles, 1.5 IQR whiskers,
    white median dot). Points beyond the whiskers are drawn and labelled with
    the object's initial (empty name = no label)."""
    parts = ax.violinplot(data, positions=[pos], widths=width, showmeans=False,
                          showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_edgecolor("black"); pc.set_linewidth(0.9)
        pc.set_linestyle("-"); pc.set_alpha(0.45)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = float(np.min(data[data >= q1 - 1.5 * iqr])), float(np.max(data[data <= q3 + 1.5 * iqr]))
    bw = width * 0.22
    ax.add_patch(plt.Rectangle((pos - bw / 2, q1), bw, iqr, linewidth=1.1, edgecolor="black",
                               linestyle="-", facecolor="white", zorder=3))
    for a, b in ((lo, q1), (q3, hi)):
        ax.plot([pos, pos], [a, b], color="black", linewidth=1.1, linestyle="-", zorder=3)
    for y in (lo, hi):
        ax.plot([pos - bw * 0.275, pos + bw * 0.275], [y, y], color="black", linewidth=1.1, zorder=3)
    ax.scatter([pos], [med], s=12, color="white", edgecolor="black", linewidth=0.6, zorder=5)
    out = (data < lo) | (data > hi)
    if not out.any():
        return
    ax.scatter(np.full(out.sum(), pos), data[out], s=9, color=color, edgecolor="black",
               linewidth=0.5, alpha=0.85, zorder=6)
    ylo, yhi = ax.get_ylim()
    clearance = 0.030 * ((yhi - ylo) if yhi > ylo else 1.0)
    placed = []
    for val, name in zip(data[out], names[out]):
        if str(name) == "":
            continue
        side = next((-ps for py, ps in placed if abs(py - val) < clearance), +1)
        placed.append((float(val), side))
        ax.text(pos + side * width * 0.38, float(val), C.short_label(name),
                fontsize=3.5, va="center", ha="left" if side > 0 else "right", color="black",
                style="italic", zorder=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))


def plot_absolute_gap_violin(gaps: pd.DataFrame, ctrls: list, styles: dict, sr_pct: float, path: Path):
    """One violin per (condition, controller); black bar + number = mean gap.
    Low-success-rate objects stay in the distribution and are marked with a
    hollow red diamond and their initial."""
    n_conds, n_ctrls = len(C.CONDITIONS), len(ctrls)
    span = 0.80
    v_width = span / n_ctrls * 0.82
    offsets = np.linspace(-span / 2 + span / (2 * n_ctrls), span / 2 - span / (2 * n_ctrls), n_ctrls)
    centres = np.arange(n_conds, dtype=float)
    fig, ax = plt.subplots(figsize=(max(2.8 + 0.5 * max(n_ctrls - 2, 0), 3.2), 2.2))
    handles, flagged = [], []
    for ki, ctrl in enumerate(ctrls):
        color = styles[ctrl]["color"]
        for xi, cond in zip(centres + offsets[ki], C.CONDITIONS):
            g = gaps[(gaps.controller == ctrl) & (gaps.condition == cond)]
            gap, names, low = g["gap_s"].values, g["object"].values.astype(object), g["low_sr"].values
            labels = names.copy()
            labels[low] = ""  # low-SR points get their label in the flag pass below
            if gap.size >= 2:
                draw_violin_box(ax, float(xi), gap, color, v_width, labels)
            elif gap.size == 1:
                ax.scatter([xi], gap, s=18, color=color, zorder=5, alpha=0.8)
            if gap.size:
                mn, half = float(np.mean(gap)), v_width * 0.28
                ax.plot([xi - half, xi + half], [mn, mn], color="black", linewidth=1.0, zorder=6,
                        solid_capstyle="butt", alpha=0.85)
                ax.text(xi + half + 0.02, mn, f"{mn:.2f}", fontsize=4.5, va="center", ha="left",
                        fontweight="bold", color="black", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.65))
            flagged += [(float(xi), float(v), str(n), color) for v, n in zip(gap[low], names[low])]
        handles.append(mpatches.Patch(facecolor=color, alpha=0.6, edgecolor="black", linewidth=0.5,
                                      label=styles[ctrl]["label"]))
    for gi in range(n_conds - 1):
        ax.axvline((centres[gi] + centres[gi + 1]) / 2, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_xticks(centres)
    ax.set_xticklabels([C.COND_LABEL[c] for c in C.CONDITIONS], rotation=20, ha="right", fontsize=C.FONT_SIZE_TICK)
    ax.set_ylabel(LABEL_GAP, fontweight="bold", fontsize=C.FONT_SIZE_LABEL)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_xlim(centres[0] - span, centres[-1] + span)
    plt.tight_layout(pad=C.TIGHT_PAD)
    legend_fs = max(C.FONT_SIZE_LEGEND - 1.5, 4.0)
    if flagged:
        ylo, yhi = ax.get_ylim()
        yr = yhi - ylo if yhi > ylo else 1.0
        placed = []
        for x, y, name, color in flagged:
            y = float(np.clip(y, ylo + 0.025 * yr, yhi - 0.025 * yr))
            ax.scatter([x], [y], s=9, color=color, edgecolor="black", linewidth=0.5, alpha=0.85,
                       zorder=6, clip_on=False)
            ax.scatter([x], [y], marker="D", s=12, facecolors="none", edgecolors="#cc6666",
                       linewidths=0.8, zorder=8, clip_on=False, alpha=0.70)
            ylbl = y
            for px, py in placed:
                if abs(px - x) < 1e-6 and abs(py - ylbl) < 0.03 * yr:
                    ylbl = py + 0.03 * yr
            placed.append((x, ylbl))
            ax.text(x + v_width * 0.30, ylbl, C.short_label(name), fontsize=3.5, va="center", ha="left",
                    style="italic", color="black", zorder=9, clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.6))
        handles.append(mlines.Line2D([], [], marker="D", color="#cc6666", linestyle="none", markersize=4,
                                     markerfacecolor="none", markeredgewidth=0.8, label=f"SR < {int(sr_pct)}%"))
    ax.legend(handles=handles, fontsize=legend_fs, loc="best", framealpha=0.85, ncol=1)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Real-world results: loading and per-object summaries
# ---------------------------------------------------------------------------
def load_realworld_csv(path, top_n_shortest, object_col="obj_name", level_col="clutter_level",
                       cost_col="object_execute_s") -> pd.DataFrame:
    """Real-world log (one row per object pick) -> (object, condition, time) rows.

    Columns default to the paper's clutter_compile.csv; the level column may use
    0-3, E0-E3 or the condition names. If top_n_shortest is set, only the N
    lowest-cost picks of each (object, clutter level) are kept (the paper uses N = 3).
    """
    df = pd.read_csv(path)
    missing = [c for c in (object_col, level_col, cost_col) if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing}; set --csv_object_col / --csv_level_col / --csv_cost_col")
    df["condition"] = df[level_col].map(C.normalize_condition)
    df = df[df["condition"].notna()].copy()
    df["object"] = df[object_col].astype(str)
    df["time"] = pd.to_numeric(df[cost_col], errors="coerce")
    df = df.dropna(subset=["time"])[["object", "condition", "time"]].reset_index(drop=True)
    if top_n_shortest:
        df = (df.sort_values("time").groupby(["object", "condition"], group_keys=False)
                .head(top_n_shortest).reset_index(drop=True))
    return df


def cell_times(trials: pd.DataFrame, obj: str, cond: str, z: float) -> np.ndarray:
    """MAD-filtered picking times of one (object, condition) cell."""
    sel = (trials["object"] == obj) & (trials["condition"] == cond)
    return C.mad_filter(trials.loc[sel, "time"].to_numpy(dtype=float), z)


def object_summary(trials: pd.DataFrame, z: float) -> pd.DataFrame:
    """Per object x condition: D (mean filtered picking time), std (population) and N."""
    rows = []
    for obj in sorted(trials["object"].unique()):
        for cond in C.CONDITIONS:
            t = cell_times(trials, obj, cond, z)
            rows.append(dict(object=obj, condition=cond, D_mean_s=float(np.mean(t)) if t.size else np.nan,
                             D_std_s=float(np.std(t)) if t.size >= 2 else np.nan, n_trials=int(t.size)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gamma GLM with log link (IRLS)
# ---------------------------------------------------------------------------
TERMS = ["intercept", "clutter_complexity", "grasp_difficulty", "clutter_x_grasp"]
TERM_LABEL = {"clutter_complexity": "Clutter severity", "grasp_difficulty": "Grasp difficulty",
              "clutter_x_grasp": "Clutter x grasp difficulty\n(interaction)"}


def glm_design(trials: pd.DataFrame, z: float, difficulty: dict = None):
    """Trial-level design matrix for the grasp-score GLM.

    grasp_score   = picking time of one successful trial (after the MAD filter)
    clutter       = 0 (isolated), 1 (E1), 2 (E2), 3 (E3)
    grasp_difficulty = the object's value in ``difficulty`` (--object_difficulty CSV);
                    by default its 1-based rank in alphabetical order of the
                    object names (EGAD names sort in their designed difficulty
                    order, e.g. A24_0 < B25_3 < ...)
    """
    objs = sorted(trials["object"].unique())
    if difficulty:
        missing = [o for o in objs if o not in difficulty]
        if missing:
            raise SystemExit(f"--object_difficulty has no value for {missing}")
        rank = {o: float(difficulty[o]) for o in objs}
    else:
        rank = {o: i + 1 for i, o in enumerate(objs)}
    clutter, diff, y = [], [], []
    for obj in objs:
        for k, cond in enumerate(C.CONDITIONS):
            for v in cell_times(trials, obj, cond, z):
                clutter.append(float(k)); diff.append(float(rank[obj])); y.append(float(v))
    clutter, diff = np.array(clutter), np.array(diff)
    X = np.column_stack([np.ones(len(y)), clutter, diff, clutter * diff])
    return X, np.array(y), len(objs)


def fit_gamma_log_glm(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> dict:
    """Gamma-family GLM with log link, fit by IRLS.

    Standard errors are the classical model-based ones,
    Cov = phi * (X'WX)^-1 with the dispersion phi estimated from Pearson
    residuals, and p-values are two-sided t-tests with n - k degrees of
    freedom (the default of R's summary.glm()).
    """
    n, k = X.shape
    y = np.maximum(y, 1e-8)
    eta = np.log(np.maximum((y + np.mean(y)) / 2.0, 1e-8))
    for _ in range(max_iter):
        mu = np.maximum(np.exp(eta), 1e-8)
        W = mu ** 2 / np.clip(mu ** 2, 1e-12, None)          # (dmu/deta)^2 / V(mu)
        zw = eta + (y - mu) / np.clip(mu, 1e-12, None)        # working response
        beta = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * zw))
        eta_new = X @ beta
        done = np.max(np.abs(eta_new - eta)) < tol
        eta = eta_new
        if done:
            break
    mu = np.maximum(np.exp(eta), 1e-8)
    V = np.clip(mu ** 2, 1e-12, None)
    W = mu ** 2 / V
    df_resid = max(n - k, 1)
    phi = max(float(np.sum(((y - mu) / np.sqrt(V)) ** 2)) / df_resid, 1e-12)
    se = np.sqrt(np.clip(np.diag(phi * np.linalg.inv(X.T @ (X * W[:, None]))), 0.0, None))
    t = beta / se
    return dict(coef=beta, se=se, t=t, p=2.0 * stats.t.sf(np.abs(t), df_resid), n_rows=n)


# ---------------------------------------------------------------------------
# Real-world figures
# ---------------------------------------------------------------------------
def load_thumbnail(thumb_dir, obj):
    """Render <thumb_dir>/<obj>.pdf to an RGBA array with white made transparent (needs PyMuPDF)."""
    if thumb_dir is None:
        return None
    path = Path(thumb_dir) / f"{obj}.pdf"
    if not path.exists():
        matches = sorted(Path(thumb_dir).glob(f"{obj}*.pdf"))
        if not matches:
            return None
        path = matches[0]
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    doc = fitz.open(str(path))
    page = doc[0]
    zoom = 600.0 / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=True)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4).copy()
    doc.close()
    img[(img[:, :, :3] >= 253).all(axis=2), 3] = 0
    return img


def auto_ylim(values):
    v = np.concatenate([np.asarray(s, float).ravel() for s in values]) if values else np.array([])
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * 0.08 if hi != lo else (1.0 if hi == 0 else abs(hi) * 0.1)
    lo2, hi2 = max(lo - pad, 0.0), hi + pad
    return (lo2, hi2 if hi2 > lo2 else lo2 + 1.0)


def plot_time_trend_overlay(summ: pd.DataFrame, line_ctrls, styles, thumb_dir, path: Path):
    """Lines (colour = condition, line style/marker = controller) with a
    +/-1 std band for each simulated controller on the left axis; real-world
    mean +/- std as stars on a second (right) axis, because real-world times
    are several times longer than simulated ones."""
    objects = sorted(summ["object"].unique())
    if thumb_dir is not None and not Path(thumb_dir).is_dir():
        print(f"[WARN] thumbnail_dir not found: {thumb_dir} - thumbnails disabled")
        thumb_dir = None
    x = np.arange(len(objects))
    xi_of = {o: i for i, o in enumerate(objects)}
    thumbs = [load_thumbnail(thumb_dir, o) for o in objects] if thumb_dir else []
    have_thumbs = thumb_dir is not None
    head = 1.22 if have_thumbs else 1.15
    fig, ax = plt.subplots(figsize=(max(3.5, 0.35 * len(objects)), 4.6 if have_thumbs else 4.0))
    band = []
    for ctrl in line_ctrls:
        for cond in C.CONDITIONS:
            s = summ[(summ.controller == ctrl) & (summ.condition == cond)].set_index("object").reindex(objects)
            y, e = s["D_mean_s"].values, s["D_std_s"].values
            ok = np.isfinite(y) & np.isfinite(e)
            if not ok.any():
                continue
            ax.plot(x[ok], y[ok], marker=styles[ctrl]["marker"], linestyle=styles[ctrl]["linestyle"],
                    linewidth=1.0, markersize=3.0, color=C.COND_COLOR[cond], alpha=0.95)
            lo, hi = np.maximum(y[ok] - e[ok], 0.0), y[ok] + e[ok]
            hi = np.maximum(hi, lo)
            ax.fill_between(x[ok], lo, hi, color=C.COND_COLOR[cond], alpha=0.12, linewidth=0)
            band += [lo, hi]
    ax.set_xlabel(r"Objects $O_i$", fontweight="bold", fontsize=11)
    ax.set_ylabel(LABEL_D + " — Simulated", fontweight="bold", fontsize=11)
    ax.yaxis.set_label_coords(-0.075, 0.38)
    ax.set_xticks(x)
    ax.set_xticklabels([C.short_label(o) for o in objects], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    real = summ[summ.controller == REAL]
    if not real.empty:
        ax2 = ax.twinx()
        rcolor = styles[REAL]["color"]
        jitter = np.linspace(-0.18, 0.18, 4)
        rband = []
        for j, cond in enumerate(C.CONDITIONS):
            s = real[real.condition == cond]
            y, e = s["D_mean_s"].values, s["D_std_s"].values
            ok = np.isfinite(y)
            if not ok.any():
                continue
            xs = np.array([xi_of[o] for o in s["object"]], float) + jitter[j]
            e = np.nan_to_num(e[ok])
            ax2.errorbar(xs[ok], y[ok], yerr=e, color=C.COND_COLOR[cond], marker="*", markeredgewidth=0.8,
                         markeredgecolor="black", markersize=15, linestyle="none", capsize=3,
                         elinewidth=1.4, zorder=6)
            rband += list(y[ok] - e) + list(y[ok] + e)
        yl = auto_ylim([rband])
        if yl is not None:
            ax2.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * head)
        ax2.set_ylabel(LABEL_D + " — Real-World", fontweight="bold",
                       fontsize=11, color=rcolor)
        ax2.yaxis.set_label_coords(1.075, 0.38)
        ax2.tick_params(axis="y", labelcolor=rcolor, labelsize=8.5)

    note = {"-": "solid", ":": "dotted", "--": "dashed", "-.": "dash-dot"}
    handles = [Line2D([0], [0], color=C.COND_COLOR[c], marker="o", linestyle="-", linewidth=1.6,
                      label=C.COND_LABEL[c]) for c in C.CONDITIONS]
    for c in line_ctrls:
        ls = styles[c]["linestyle"]
        lab = styles[c]["label"] + (f" ({note[ls]} line)" if isinstance(ls, str) and ls in note else "")
        handles.append(Line2D([0], [0], color="black", marker=styles[c]["marker"], markeredgewidth=0,
                              markersize=7, linestyle=ls, linewidth=1.6, label=lab))
    if not real.empty:
        handles.append(Line2D([0], [0], color="black", marker="*", markeredgewidth=0.8, markersize=11,
                              linestyle="none", linewidth=1.6, label=f"{styles[REAL]['label']} (points only)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), fontsize=8.5, ncol=4,
              handlelength=2.2, frameon=False, columnspacing=1.0)
    yl = auto_ylim(band)
    if yl is not None:
        ax.set_ylim(yl[0], yl[0] + (yl[1] - yl[0]) * head)

    if any(img is not None for img in thumbs):
        blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        target_in = fig.get_figwidth() * ax.get_position().width / len(objects) * 1.10
        for xi, img in enumerate(thumbs):
            if img is None:
                continue
            oi = OffsetImage(img, zoom=(target_in * 100) / max(img.shape[:2]), interpolation="lanczos")
            oi.image.axes = ax
            ax.add_artist(AnnotationBbox(oi, (xi, 0.80), xycoords=blended, box_alignment=(0.5, 0.5),
                                         frameon=False, pad=0, zorder=5))
    plt.tight_layout(pad=C.TIGHT_PAD)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.5)
    plt.close()


def plot_coefficient_grid(coefs: pd.DataFrame, ctrls, styles, path: Path):
    """One group per GLM term, one bar per controller; whiskers = 95% CI
    (1.96 SE); faded bar = not significant (p >= .05); label = value + stars."""
    terms = TERMS[1:]
    n = len(ctrls)
    bar_w = 1.35 / n
    gap = n * bar_w + 0.35
    centers = np.arange(len(terms)) * gap
    offs = (np.arange(n) - (n - 1) / 2.0) * bar_w
    fig, ax = plt.subplots(figsize=(2.45 * len(terms) + 0.6, 3.6))
    for ti, term in enumerate(terms):
        sub = coefs[coefs.term == term].set_index("controller").loc[ctrls]
        b, ci, p, sig = sub["coef"].values, 1.96 * sub["se"].values, sub["p"].values, sub["significant"].values
        xs = centers[ti] + offs
        for xpos, bv, s, c in zip(xs, b, sig, ctrls):
            ax.bar(xpos, bv, width=bar_w * 0.92, color=C.lighten(styles[c]["color"], 0.55 if s else 0.80),
                   edgecolor="black" if s else "#aaaaaa", linewidth=0.6, zorder=3)
        ax.errorbar(xs, b, yerr=ci, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=0, alpha=0.65, zorder=4)
        for xpos, bv, civ, pv in zip(xs, b, ci, p):
            d = 1 if bv >= 0 else -1
            stars, val = f"({C.sig_stars(pv)})", f"{bv:.2g}"
            ax.annotate(f"{stars}\n{val}" if d > 0 else f"{val}\n{stars}", xy=(xpos, bv + civ * d),
                        xytext=(0, 3.0 * d), textcoords="offset points", ha="center",
                        va="bottom" if d > 0 else "top", fontsize=5.5, fontweight="bold", color="black",
                        linespacing=1.05, zorder=5)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_xticks(centers)
    ax.set_xticklabels([TERM_LABEL[t] for t in terms], fontsize=9, fontweight="bold")
    for tc in centers[:-1]:
        ax.axvline(tc + gap / 2, color="gray", linewidth=0.5, alpha=0.35, zorder=1)
    pad = n * bar_w / 2 + 0.25
    ax.set_xlim(centers.min() - pad, centers.max() + pad)
    fig.canvas.draw()
    yt = ax.get_yticks()
    step = abs(yt[1] - yt[0]) if len(yt) > 1 else 0
    dec = ((max(0, -int(np.floor(np.log10(step)))) if step > 0 else 1) + 1) if len(yt) > 1 else 2
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p, d=dec: f"{v:.{d}f}"))
    ax.tick_params(axis="y", labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ylo, yhi = ax.get_ylim()
    yr = (yhi - ylo) if yhi > ylo else 1.0
    ax.set_ylim(ylo - 0.14 * yr, yhi + 0.14 * yr)
    handles = [mpatches.Patch(facecolor=styles[c]["color"], edgecolor="black", linewidth=0.7,
                              label=styles[c]["label"]) for c in ctrls]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="black", alpha=0.30, linewidth=0.7,
                                  label="Faded = not sig. (p>=.05)"))
    ax.legend(handles=handles, fontsize=10, loc="upper right", framealpha=0.9, ncol=1, handlelength=1.6,
              borderaxespad=0.3, labelspacing=0.4)
    ax.set_ylabel("Coefficient", fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(pad=0.4)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
def run_sim(runs: dict, ref: str, styles: dict, args, out: Path):
    """Figures 1-2 from the simulated controllers."""
    metrics = {name: object_metrics(r, args.outlier_z) for name, r in runs.items()}
    for name, m in metrics.items():
        if m.empty or not np.isfinite(m["D_isolated"].values).any():
            trial = next((t for conds in runs[name].values() for d in conds.values()
                          for t in d.get("trial_results", []) if t), {})
            raise SystemExit(
                f"{name}: no usable isolated-condition costs. Every object needs an isolated run, and each "
                f"trial needs a numeric cost field (looked for {[C.COST_KEY] if C.COST_KEY else list(C.TIME_KEYS)}"
                f"{'' if C.ALL_TRIALS else ' in successful trials'}). A trial in this folder has keys "
                f"{sorted(trial)}; set --cost KEY (and --all_trials if the cost applies to failed trials).")
    pd.concat(metrics, names=["controller"]).to_csv(out / "object_metrics.csv")
    rt = ranking_table(metrics[ref])
    rt.to_csv(out / f"ranking_heatmap_{ref}.csv", index=False)
    plot_ranking_heatmap(rt, out / f"ranking_heatmap_{ref}.pdf")
    gaps = gap_table(metrics, ref, args.sr_threshold / 100.0)
    gaps.to_csv(out / "absolute_gap_violin_all_ctrls.csv", index=False)
    summary = (gaps.groupby(["controller", "condition"], sort=False)["gap_s"]
               .agg(n_objects="size", mean="mean", median="median", std=lambda s: s.std(ddof=1)).reset_index())
    summary.to_csv(out / "absolute_gap_summary.csv", index=False)
    plot_absolute_gap_violin(gaps, list(runs), styles, args.sr_threshold, out / "absolute_gap_violin_all_ctrls.pdf")


def run_realworld(runs: dict, styles: dict, args, out: Path):
    """Figures 3-4: real-world trends (+ simulated --trend lines) and the per-controller GLM."""
    trials = {name: C.trials_table(r) for name, r in runs.items()}
    real = load_realworld_csv(args.csv, args.top_n_shortest or None,
                              args.csv_object_col, args.csv_level_col, args.csv_cost_col)
    if args.objects:
        real = real[real["object"].isin(args.objects)].reset_index(drop=True)
    print(f"Real-world: {len(real)} picks, objects {sorted(real['object'].unique())}")

    # Figure 3: per-object trends (each controller on its own full object set)
    summ = pd.concat([object_summary(trials[c], args.outlier_z).assign(controller=c) for c in args.trend]
                     + [object_summary(real, args.outlier_z).assign(controller=REAL)], ignore_index=True)
    summ = summ[["controller", "object", "condition", "D_mean_s", "D_std_s", "n_trials"]]
    summ.to_csv(out / "time_trend_overlay.csv", index=False)
    plot_time_trend_overlay(summ, args.trend, styles, args.thumbnail_dir, out / "time_trend_overlay.pdf")

    # Figure 4: full GLM per simulated controller (real-world excluded: too few objects)
    if not trials:
        print("No --controller given: skipping coef_all_terms (the GLM is fit on simulated controllers).")
        return
    rows = []
    for c, tr in trials.items():
        X, y, n_obj = glm_design(tr, args.outlier_z, args.difficulty)
        fit = fit_gamma_log_glm(X, y)
        for i, term in enumerate(TERMS[1:], start=1):
            rows.append(dict(controller=c, controller_label=styles[c]["label"], term=term, coef=fit["coef"][i],
                             se=fit["se"][i], t=fit["t"][i], p=fit["p"][i], significant=bool(fit["p"][i] < 0.05),
                             n_rows=fit["n_rows"], n_objects=n_obj))
        print(f"GLM {c}: n={fit['n_rows']} trials, {n_obj} objects")
    coefs = pd.DataFrame(rows)
    coefs.to_csv(out / "coef_all_terms.csv", index=False)
    plot_coefficient_grid(coefs, list(trials), styles, out / "coef_all_terms.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", action="append", default=[], metavar="NAME=DIR",
                    help="Simulated controller and its folder of run directories (repeatable; order = plot order).")
    ap.add_argument("--csv", default=None, help="Real-world clutter_compile.csv (enables the real-world figures).")
    ap.add_argument("--reference", default=None,
                    help="Controller for the ranking heatmap and the gap baseline (default: first --controller).")
    ap.add_argument("--trend", nargs="*", default=None, metavar="NAME",
                    help="Simulated controllers drawn as lines in time_trend_overlay "
                         "(default: rl and heuristic if given, else the first two --controller names).")
    ap.add_argument("--label", action="append", metavar="NAME=TEXT", help="Display label override.")
    ap.add_argument("--color", action="append", metavar="NAME=COLOR", help="Colour override, e.g. rl=#e8c84a.")
    ap.add_argument("--prefix", default="single", help="Run-folder name prefix (default: single).")
    ap.add_argument("--sr_threshold", type=float, default=60.0,
                    help="Success rate (%%) below which an object is flagged in the gap figure (default 60).")
    ap.add_argument("--outlier_z", type=float, default=3.5,
                    help="Robust-z threshold of the trial outlier filter; 0 disables it (default 3.5).")
    ap.add_argument("--top_n_shortest", type=int, default=3,
                    help="Keep the N fastest real-world picks per (object, level); 0 keeps all (default 3).")
    ap.add_argument("--thumbnail_dir", default=None, help="Optional folder of <object>.pdf images (needs PyMuPDF).")
    g = ap.add_argument_group("protocol settings for your own data (defaults reproduce the paper)")
    g.add_argument("--cost", default=None, metavar="KEY",
                   help="Per-trial cost field C(tau) in trial_results (default: picking_time).")
    g.add_argument("--all_trials", action="store_true",
                   help="Average the cost over all trials, not only successful ones (e.g. for energy).")
    g.add_argument("--cost_label", default=None, metavar="TEXT",
                   help="Name of the cost in axis labels, e.g. 'Energy (J)' (default: picking-time labels).")
    g.add_argument("--objects", nargs="+", default=None, metavar="OBJ",
                   help="Object set O to analyse (default: every object with an isolated run).")
    g.add_argument("--object_difficulty", default=None, metavar="CSV",
                   help="CSV with columns object,difficulty for the GLM's object-difficulty term "
                        "(default: alphabetical rank of the object names, as for EGAD).")
    g.add_argument("--csv_object_col", default="obj_name", help="Object column of --csv (default obj_name).")
    g.add_argument("--csv_level_col", default="clutter_level",
                   help="Clutter-level column of --csv: 0-3, E0-E3 or condition names (default clutter_level).")
    g.add_argument("--csv_cost_col", default="object_execute_s", help="Cost column of --csv (default object_execute_s).")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    dirs = C.parse_key_values(args.controller, "--controller")
    if not dirs and not args.csv:
        ap.error("give at least one --controller NAME=DIR and/or --csv")
    ref = args.reference or next(iter(dirs), None)
    if dirs and ref not in dirs:
        raise SystemExit(f"--reference {ref!r} is not one of the controllers {list(dirs)}")
    if args.trend is None:
        args.trend = [c for c in ("rl", "heuristic") if c in dirs] or list(dirs)[:2]
    unknown = [c for c in args.trend if c not in dirs]
    if unknown:
        raise SystemExit(f"--trend names {unknown} are not --controller names {list(dirs)}")
    styles = C.build_styles(list(dirs) + [REAL], C.parse_key_values(args.label, "--label"),
                            C.parse_key_values(args.color, "--color"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    C.apply_paper_style()
    C.COST_KEY, C.ALL_TRIALS = args.cost, args.all_trials
    if args.cost_label:
        set_cost_label(args.cost_label)
    args.difficulty = None
    if args.object_difficulty:
        dt = pd.read_csv(args.object_difficulty)
        args.difficulty = dict(zip(dt["object"].astype(str), dt["difficulty"]))

    runs = {}
    for name, d in dirs.items():
        runs[name] = C.load_runs(d, args.prefix)
        if args.objects:
            runs[name] = {o: r for o, r in runs[name].items() if o in set(args.objects)}
            if not runs[name]:
                raise SystemExit(f"{name}: none of --objects {args.objects} found in {d}")
        print(f"Loaded {name}: {len(runs[name])} objects from {d}")
    if runs:
        run_sim(runs, ref, styles, args, out)
    if args.csv:
        run_realworld(runs, styles, args, out)
    print(f"Wrote figures and CSVs to {out.resolve()}")


if __name__ == "__main__":
    main()
