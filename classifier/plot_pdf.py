"""
Plot All Results as PDF
=======================
Loads exactly what the two pipeline scripts save to disk:

pca_clustering.py saves (data/classifier/pca_kmeans/):
  - pca_loadings.csv        → scree + loadings heatmap + contribution bars
  - pca_results.csv         → plain PCA scatter plots
  - pca_kmeans_results.csv  → cluster-coloured scatter plots
  - kmeans_metrics.csv      → silhouette / CH score summary table

training.py saves (data/classifier/training/):
  - history.npy             → all 6 training curve panels
  - results.txt             → classification report table
  - test_predictions.npy / test_labels.npy → confusion matrix + report

Usage:
  python plot_pdf.py
  python plot_pdf.py --pca_dir X --cnn_dir Y --data_dir Z
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from itertools import combinations

# Repository root (classifier/ -> repo) and git-ignored data directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CLF_DATA = DATA_DIR / "classifier"


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_pdf(fig, path: Path):
    ensure_dir(path.parent)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ──────────────────────────────────────────────────────────
# 1. PCA PLOTS  (pca_loadings.csv + pca_kmeans_results.csv)
# ──────────────────────────────────────────────────────────
def plot_pca(pca_dir: Path, out_dir: Path):
    print("\n" + "="*60)
    print("1. PCA / KMeans Plots")
    print("="*60)
    ensure_dir(out_dir)

    loadings_csv = pca_dir / "pca_loadings.csv"
    if not loadings_csv.exists():
        print(f"  ⚠ pca_loadings.csv not found in {pca_dir}"); return

    df_load = pd.read_csv(loadings_csv, index_col=0)

    # Pipeline writes a "Variance %" row at the top
    if "Variance %" in df_load.index:
        variance    = df_load.loc["Variance %"].values.astype(float)
        loadings_df = df_load.drop(index="Variance %").astype(float)
    else:
        variance    = None
        loadings_df = df_load.astype(float)

    pc_names = loadings_df.columns.tolist()

    # ── Scree ──────────────────────────────────
    if variance is not None:
        cumulative = np.cumsum(variance)
        x = np.arange(1, len(variance) + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x, variance, alpha=0.6, label="Individual")
        ax.plot(x, cumulative, "ro-", linewidth=2, markersize=6, label="Cumulative")
        for i, v in enumerate(variance):
            ax.text(i + 1, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
        ax.set_xlabel("Principal Component", fontsize=11)
        ax.set_ylabel("Explained Variance (%)", fontsize=11)
        ax.set_title("Scree Plot – Explained Variance by Component",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        save_pdf(fig, out_dir / "pca_scree.pdf")

    # ── Loadings heatmap ──────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(pc_names) * 1.2),
                                    max(4, len(loadings_df) * 0.7)))
    sns.heatmap(loadings_df, annot=True, fmt=".3f", cmap="vlag", center=0,
                ax=ax, cbar_kws={"label": "Loading Weight"})
    ax.set_title("PCA Loadings (Features × Components)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Principal Components", fontsize=11)
    ax.set_ylabel("Features", fontsize=11)
    save_pdf(fig, out_dir / "pca_loadings_heatmap.pdf")

    # ── Contribution bars (one per PC) ────────
    for pc_idx, pc_name in enumerate(pc_names):
        series = loadings_df[pc_name].sort_values(key=lambda x: np.abs(x), ascending=False)
        colors = ["green" if v > 0 else "red" for v in series.values]
        var_str = f" ({variance[pc_idx]:.1f}% var)" if variance is not None else ""
        fig, ax = plt.subplots(figsize=(10, 5))
        series.plot(kind="barh", ax=ax, color=colors, alpha=0.7)
        ax.set_xlabel("Loading Weight", fontsize=11)
        ax.set_title(f"Feature Contributions to {pc_name}{var_str}",
                     fontsize=11, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="x")
        save_pdf(fig, out_dir / f"pca_{pc_name.lower()}_contributions.pdf")

    # ── Scatter plots ─────────────────────────
    results_csv = pca_dir / "pca_kmeans_results.csv"
    if not results_csv.exists():
        results_csv = pca_dir / "pca_results.csv"
    if not results_csv.exists():
        print(f"  ⚠ No results CSV found for scatter plots"); return

    df = pd.read_csv(results_csv)
    pc_cols = [c for c in df.columns if re.fullmatch(r"PC\d+", c)]
    has_clusters = "kmeans_cluster" in df.columns
    palette = sns.color_palette("tab10", n_colors=10)

    for pc_x, pc_y in combinations(pc_cols[:4], 2):
        ncols = 2 if has_clusters else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
        if ncols == 1: axes = [axes]

        axes[0].scatter(df[pc_x], df[pc_y], s=20, alpha=0.5, color="steelblue")
        axes[0].set_xlabel(pc_x, fontsize=11); axes[0].set_ylabel(pc_y, fontsize=11)
        axes[0].set_title(f"PCA: {pc_x} vs {pc_y}", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        if has_clusters:
            for i, cid in enumerate(sorted(df["kmeans_cluster"].unique())):
                mask = df["kmeans_cluster"] == cid
                axes[1].scatter(df.loc[mask, pc_x], df.loc[mask, pc_y],
                                s=20, alpha=0.6, label=f"Cluster {cid}",
                                color=palette[i % 10])
            axes[1].set_xlabel(pc_x, fontsize=11); axes[1].set_ylabel(pc_y, fontsize=11)
            axes[1].set_title(f"KMeans Clusters: {pc_x} vs {pc_y}",
                              fontsize=12, fontweight="bold")
            axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_pdf(fig, out_dir / f"scatter_{pc_x}_vs_{pc_y}.pdf")

    # ── KMeans metrics table ──────────────────
    metrics_csv = pca_dir / "kmeans_metrics.csv"
    if metrics_csv.exists():
        df_m = pd.read_csv(metrics_csv)
        fig, ax = plt.subplots(figsize=(8, max(2, len(df_m) * 0.6 + 1)))
        ax.axis("off")
        tbl = ax.table(cellText=df_m.values, colLabels=df_m.columns,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.3, 1.6)
        ax.set_title("KMeans Clustering Metrics", fontsize=13, fontweight="bold", pad=12)
        save_pdf(fig, out_dir / "kmeans_metrics_table.pdf")


# ──────────────────────────────────────────────────────────
# 2. TRAINING CURVES  (history.npy)
# ──────────────────────────────────────────────────────────
def plot_training(cnn_dir: Path, out_dir: Path):
    print("\n" + "="*60)
    print("2. Training Curves")
    print("="*60)
    ensure_dir(out_dir)

    history_path = cnn_dir / "history.npy"
    if not history_path.exists():
        print(f"  ⚠ history.npy not found in {cnn_dir}"); return

    h          = np.load(history_path, allow_pickle=True).item()
    train_loss = h["train_loss"]; val_loss = h["val_loss"]
    train_acc  = h["train_acc"];  val_acc  = h["val_acc"]
    epochs     = range(1, len(train_loss) + 1)
    best_ep    = int(np.argmax(val_acc))
    best_acc   = val_acc[best_ep]
    colors     = ["#3498db", "#e74c3c"]

    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Loss curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, train_loss, "b-o", label="Train Loss", linewidth=2, markersize=3)
    ax.plot(epochs, val_loss,   "r-s", label="Val Loss",   linewidth=2, markersize=3)
    ax.axvline(best_ep + 1, color="green", linestyle="--", linewidth=1.5,
               label=f"Best Epoch ({best_ep+1})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, train_acc, "b-o", label="Train Acc", linewidth=2, markersize=3)
    ax.plot(epochs, val_acc,   "r-s", label="Val Acc",   linewidth=2, markersize=3)
    ax.axvline(best_ep + 1, color="green", linestyle="--", linewidth=1.5,
               label=f"Best Epoch ({best_ep+1})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Loss gap
    ax = fig.add_subplot(gs[1, 0])
    gap = np.array(val_loss) - np.array(train_loss)
    ax.plot(epochs, gap, "purple", linewidth=2, marker="o", markersize=3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.fill_between(epochs, 0, gap, where=gap > 0,  color="red",   alpha=0.25, label="Overfitting")
    ax.fill_between(epochs, 0, gap, where=gap <= 0, color="green", alpha=0.25, label="Underfitting")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss − Train Loss")
    ax.set_title("Overfitting Gap (Loss)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy gap
    ax = fig.add_subplot(gs[1, 1])
    gap = np.array(train_acc) - np.array(val_acc)
    ax.plot(epochs, gap, "orange", linewidth=2, marker="o", markersize=3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.fill_between(epochs, 0, gap, where=gap > 0,  color="red",   alpha=0.25, label="Overfitting")
    ax.fill_between(epochs, 0, gap, where=gap <= 0, color="green", alpha=0.25, label="Underfitting")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Acc − Val Acc (%)")
    ax.set_title("Overfitting Gap (Accuracy)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Final loss bar
    ax = fig.add_subplot(gs[2, 0])
    vals = [train_loss[-1], val_loss[-1]]
    bars = ax.bar(["Train Loss", "Val Loss"], vals, color=colors, alpha=0.75, edgecolor="black", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Loss"); ax.set_title("Final Epoch Loss Comparison", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Final accuracy bar
    ax = fig.add_subplot(gs[2, 1])
    vals = [train_acc[-1], val_acc[-1]]
    bars = ax.bar(["Train Acc", "Val Acc"], vals, color=colors, alpha=0.75, edgecolor="black", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Final Epoch Accuracy Comparison", fontweight="bold")
    ax.set_ylim([0, 105]); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Training Summary — Best Val Acc: {best_acc:.2f}%  (Epoch {best_ep+1})",
        fontsize=15, fontweight="bold", y=1.01
    )
    save_pdf(fig, out_dir / "training_curves.pdf")


# ──────────────────────────────────────────────────────────
# 3. CONFUSION MATRIX + CLASSIFICATION REPORT
#    Reads results.txt for the report.
#    Needs test_predictions.npy + test_labels.npy for the matrix.
# ──────────────────────────────────────────────────────────
def plot_evaluation(cnn_dir: Path, data_dir: Path, out_dir: Path):
    print("\n" + "="*60)
    print("3. Confusion Matrix & Classification Report")
    print("="*60)
    ensure_dir(out_dir)

    preds_path  = cnn_dir / "test_predictions.npy"
    labels_path = cnn_dir / "test_labels.npy"

    if preds_path.exists() and labels_path.exists():
        all_preds  = np.load(preds_path)
        all_labels = np.load(labels_path)

        # Infer class names from cnn_dataset/test/cluster_X dirs
        test_split = data_dir / "test"
        if test_split.exists():
            class_names = sorted([d.name for d in test_split.iterdir() if d.is_dir()])
        else:
            class_names = [str(i) for i in sorted(np.unique(all_labels))]

        test_acc = accuracy_score(all_labels, all_preds)
        cm       = confusion_matrix(all_labels, all_preds)
        cm_norm  = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title(f"Confusion Matrix  (Test Acc: {test_acc*100:.2f}%)", fontweight="bold")
        axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title("Normalised Confusion Matrix", fontweight="bold")
        axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")

        plt.tight_layout()
        save_pdf(fig, out_dir / "confusion_matrix.pdf")

        # Classification report as figure table
        report = classification_report(all_labels, all_preds,
                                       target_names=class_names, output_dict=True)
        df_rep = pd.DataFrame(report).transpose().round(4)
        fig, ax = plt.subplots(figsize=(11, max(3, len(df_rep) * 0.55 + 1)))
        ax.axis("off")
        tbl = ax.table(cellText=df_rep.values, colLabels=df_rep.columns,
                       rowLabels=df_rep.index, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.5)
        ax.set_title("Classification Report", fontsize=13, fontweight="bold", pad=12)
        save_pdf(fig, out_dir / "classification_report.pdf")

    else:
        # Fall back: render results.txt as a text figure
        results_txt = cnn_dir / "results.txt"
        if results_txt.exists():
            text = results_txt.read_text()
            fig, ax = plt.subplots(figsize=(10, max(4, text.count("\n") * 0.28)))
            ax.axis("off")
            ax.text(0.01, 0.99, text, transform=ax.transAxes,
                    fontsize=9, verticalalignment="top", fontfamily="monospace")
            ax.set_title("results.txt", fontsize=13, fontweight="bold")
            save_pdf(fig, out_dir / "results_text.pdf")
            print("  ✓ Rendered results.txt as PDF (no confusion matrix — see note below)")
        else:
            print(f"  ⚠ Neither test_predictions.npy nor results.txt found in {cnn_dir}")

        print("\n  ── To enable the confusion matrix, add after your test loop: ──")
        print("  np.save(os.path.join(config['output_dir'], 'test_predictions.npy'), np.array(all_preds))")
        print("  np.save(os.path.join(config['output_dir'], 'test_labels.npy'),      np.array(all_labels))")


# ──────────────────────────────────────────────────────────
# 4. CLUSTER SAMPLE IMAGES  (cnn_dataset/train|val|test)
# ──────────────────────────────────────────────────────────
def plot_cluster_samples(data_dir: Path, out_dir: Path, n_samples: int = 20):
    print("\n" + "="*60)
    print("4. Cluster Sample Images")
    print("="*60)
    ensure_dir(out_dir)

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  ⚠ {split_dir} not found, skipping"); continue

        cluster_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        if not cluster_dirs:
            print(f"  ⚠ No cluster subdirs in {split_dir}"); continue

        ncols = min(n_samples, 10)
        nrows = len(cluster_dirs)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.8))

        # Normalise axes to always be 2-D list
        if nrows == 1 and ncols == 1: axes = [[axes]]
        elif nrows == 1:              axes = [list(axes)]
        elif ncols == 1:              axes = [[ax] for ax in axes]
        else:                         axes = [list(row) for row in axes]

        fig.suptitle(f"Sample Images — {split.capitalize()} Split",
                     fontsize=14, fontweight="bold")

        for r, cdir in enumerate(cluster_dirs):
            imgs = sorted(list(cdir.glob("*.png")) +
                          list(cdir.glob("*.jpg")) +
                          list(cdir.glob("*.jpeg")))[:n_samples]
            for c in range(ncols):
                ax = axes[r][c]
                ax.axis("off")
                if c < len(imgs):
                    try:
                        ax.imshow(Image.open(imgs[c]).convert("L"),
                                  cmap="gray", aspect="auto")
                    except Exception:
                        ax.text(0.5, 0.5, "ERR", ha="center", va="center",
                                transform=ax.transAxes, fontsize=7)
                if c == 0:
                    ax.set_ylabel(cdir.name, fontsize=9, rotation=0,
                                  labelpad=45, va="center")
                    ax.yaxis.set_label_coords(-0.55, 0.5)

        plt.tight_layout()
        save_pdf(fig, out_dir / f"cluster_samples_{split}.pdf")


# ──────────────────────────────────────────────────────────
# MERGE ALL PDFs
# ──────────────────────────────────────────────────────────
def merge_pdfs(out_dir: Path):
    try:
        from pypdf import PdfWriter
    except ImportError:
        print("\n  ℹ  pip install pypdf  to get a single merged PDF"); return

    writer = PdfWriter()
    for p in sorted(out_dir.rglob("*.pdf")):
        writer.append(str(p))
    merged = out_dir / "all_results_combined.pdf"
    with open(merged, "wb") as f:
        writer.write(f)
    print(f"\n  ✓ Merged PDF: {merged}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main(args):
    pca_dir  = Path(args.pca_dir)
    cnn_dir  = Path(args.cnn_dir)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    print("\n" + "="*60)
    print("PLOT ALL RESULTS AS PDF")
    print("="*60)
    print(f"  PCA/KMeans dir : {pca_dir}")
    print(f"  CNN results dir: {cnn_dir}")
    print(f"  Dataset dir    : {data_dir}")
    print(f"  Output dir     : {out_dir}")

    plot_pca(pca_dir,   out_dir / "pca_plots")
    plot_training(cnn_dir, out_dir / "training_plots")
    plot_evaluation(cnn_dir, data_dir, out_dir / "evaluation_plots")
    plot_cluster_samples(data_dir, out_dir / "cluster_samples", n_samples=args.n_samples)
    merge_pdfs(out_dir)

    print("\n" + "="*60)
    print(f"DONE — PDFs saved to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_dir",   default=str(CLF_DATA / "pca_kmeans"),
                        help="PCA/KMeans output directory")
    parser.add_argument("--cnn_dir",   default=str(CLF_DATA / "training"),
                        help="CNN training output directory")
    parser.add_argument("--data_dir",  default=str(CLF_DATA / "cnn_dataset"),
                        help="CNN dataset directory (with train/val/test subdirs)")
    parser.add_argument("--out_dir",   default=str(CLF_DATA / "pdf_results"),
                        help="Where to save all PDFs")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Sample images shown per cluster row")
    main(parser.parse_args())