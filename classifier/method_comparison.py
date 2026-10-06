"""
Advanced Clustering Comparison Pipeline
Compares: KMeans, GMM, DBSCAN, HDBSCAN, Hierarchical
Stage 1: PCA exploration
Stage 2: Compare multiple clustering methods
Stage 3: CNN dataset generation with best method
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Repository root (classifier/ -> repo) and git-ignored data directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Try to import HDBSCAN (optional)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠ HDBSCAN not available. Install with: pip install hdbscan")


# ================================================
# UTIL FUNCTIONS
# ================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_figure(fig, out_path):
    out_path = Path(out_path).with_suffix('.pdf')
    ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches='tight', format='pdf')
    plt.close(fig)


# ================================================
# FEATURE SELECTION
# ================================================
def select_features_interactive(df, default_features):
    """Interactive feature selection."""
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nAvailable numeric features ({len(numeric_cols)}):")
    for i, col in enumerate(numeric_cols, 1):
        default_marker = " (DEFAULT)" if col in default_features else ""
        print(f"  {i:2d}. {col}{default_marker}")
    
    print("\nOptions:")
    print("  1. Press ENTER to use default features")
    print("  2. Type feature numbers separated by commas (e.g., 1,3,5,7)")
    print("  3. Type 'all' to use all numeric features")
    
    choice = input("\nYour choice: ").strip()
    
    if not choice:
        selected = [f for f in default_features if f in numeric_cols]
        print(f"\n✓ Using {len(selected)} default features")
        return selected
    
    if choice.lower() == 'all':
        print(f"\n✓ Using all {len(numeric_cols)} numeric features")
        return numeric_cols
    
    # Parse input
    indices = [int(x.strip())-1 for x in choice.split(',')]
    selected = [numeric_cols[i] for i in indices if 0 <= i < len(numeric_cols)]
    
    print(f"\n✓ Selected {len(selected)} features: {', '.join(selected)}")
    return selected


def select_features_from_args(df, args_features, default_features):
    """Select features from command line arguments."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if args_features:
        selected = [f for f in args_features if f in numeric_cols]
        print(f"✓ Using {len(selected)} features from arguments")
    else:
        selected = [f for f in default_features if f in numeric_cols]
        print(f"✓ Using {len(selected)} default features")
    
    return selected


# ================================================
# PCA FUNCTIONS
# ================================================
def compute_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def plot_scree(pca, out_path):
    fig, ax = plt.subplots(figsize=(8,5))
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    
    x = np.arange(1, len(ratios)+1)
    ax.bar(x, ratios*100, alpha=0.6, label='Individual')
    ax.plot(x, cumulative*100, 'ro-', linewidth=2, markersize=6, label='Cumulative')
    
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Explained Variance (%)', fontsize=11)
    ax.set_title('Scree Plot - Explained Variance by Component', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(ratios*100):
        ax.text(i+1, v+1, f'{v:.1f}%', ha='center', fontsize=8)
    
    save_figure(fig, out_path)

def scatter_pca(X_pca, pc_x, pc_y, labels=None, title='PCA scatter', out_path=None, pca=None, method_name=''):
    """Plot scatter of any two PCs."""
    fig, ax = plt.subplots(figsize=(8,7))
    
    variance_x = pca.explained_variance_ratio_[pc_x]*100 if pca else 0
    variance_y = pca.explained_variance_ratio_[pc_y]*100 if pca else 0
    
    if labels is None:
        ax.scatter(X_pca[:,pc_x], X_pca[:,pc_y], s=25, alpha=0.6, c='steelblue')
    else:
        # Handle outliers (label -1 for DBSCAN/HDBSCAN)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        palette = sns.color_palette("tab10", n_colors=max(10, n_clusters))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Outliers in black
                mask = labels == label
                ax.scatter(X_pca[mask,pc_x], X_pca[mask,pc_y], s=20, alpha=0.4, 
                          c='black', marker='x', label='Outliers')
            else:
                mask = labels == label
                ax.scatter(X_pca[mask,pc_x], X_pca[mask,pc_y], s=30, alpha=0.7, 
                          label=f'Cluster {label}', color=palette[label % len(palette)])
        
        ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    xlabel = f'PC{pc_x+1} ({variance_x:.1f}% variance)'
    ylabel = f'PC{pc_y+1} ({variance_y:.1f}% variance)'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path)


# ================================================
# CLUSTERING METHODS
# ================================================

def cluster_kmeans(X, n_clusters=3, random_seed=42):
    """KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, {'centroids': kmeans.cluster_centers_, 'inertia': kmeans.inertia_}


def cluster_gmm(X, n_clusters=3, random_seed=42):
    """Gaussian Mixture Model clustering."""
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_seed, n_init=10)
    labels = gmm.fit_predict(X)
    probabilities = gmm.predict_proba(X)
    return labels, {'bic': gmm.bic(X), 'aic': gmm.aic(X), 'probabilities': probabilities}


def cluster_dbscan(X, eps=0.5, min_samples=5):
    """DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    return labels, {'n_clusters': n_clusters, 'n_outliers': n_outliers, 'eps': eps}


def cluster_hdbscan(X, min_cluster_size=10, min_samples=5):
    """HDBSCAN clustering."""
    if not HDBSCAN_AVAILABLE:
        return None, {'error': 'HDBSCAN not installed'}
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    return labels, {
        'n_clusters': n_clusters, 
        'n_outliers': n_outliers,
        'probabilities': clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
    }


def cluster_hierarchical(X, n_clusters=3, linkage='ward'):
    """Hierarchical/Agglomerative clustering."""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)
    return labels, {'n_clusters': n_clusters, 'linkage': linkage}


def reassign_labels_by_position(X_pca, selected_pcs, labels, target_n_clusters=3):
    """
    Reassign labels based on spatial position.
    - Cluster 0: leftmost (lowest PC1)
    - Cluster 1: right-bottom (high PC1, low PC2)
    - Cluster 2: right-top (high PC1, high PC2)
    """
    unique_labels = np.unique(labels[labels >= 0])  # Exclude outliers (-1)
    n_clusters = len(unique_labels)
    
    if n_clusters != target_n_clusters:
        # Can't reassign if number doesn't match
        return labels
    
    # Calculate centroids
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = X_pca[mask][:, selected_pcs].mean(axis=0)
        centroids.append((label, centroid))
    
    pc1_idx = 0
    pc2_idx = 1 if len(selected_pcs) > 1 else 0
    
    label_mapping = {}
    
    if n_clusters == 3:
        # Leftmost -> 0
        leftmost = min(centroids, key=lambda x: x[1][pc1_idx])
        label_mapping[leftmost[0]] = 0
        
        # Among remaining, right-bottom -> 1, right-top -> 2
        remaining = [c for c in centroids if c[0] != leftmost[0]]
        if len(selected_pcs) > 1:
            right_bottom = max(remaining, key=lambda x: (x[1][pc1_idx], -x[1][pc2_idx]))
        else:
            right_bottom = max(remaining, key=lambda x: x[1][pc1_idx])
        label_mapping[right_bottom[0]] = 1
        
        last = [c for c in remaining if c[0] != right_bottom[0]][0]
        label_mapping[last[0]] = 2
    else:
        # For other numbers, left-to-right
        sorted_centroids = sorted(centroids, key=lambda x: x[1][pc1_idx])
        for new_label, (old_label, _) in enumerate(sorted_centroids):
            label_mapping[old_label] = new_label
    
    # Apply mapping (keep outliers as -1)
    new_labels = np.array([label_mapping.get(old, old) for old in labels])
    return new_labels


def compute_metrics(X, labels):
    """Compute clustering quality metrics."""
    # Filter out outliers for metrics
    mask = labels >= 0
    if np.sum(mask) < 2:
        return {'error': 'Not enough non-outlier points'}
    
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    n_clusters = len(np.unique(labels_filtered))
    if n_clusters < 2:
        return {'n_clusters': n_clusters, 'error': 'Need at least 2 clusters'}
    
    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
    except:
        metrics['silhouette'] = np.nan
    
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
    except:
        metrics['calinski_harabasz'] = np.nan
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X_filtered, labels_filtered)
    except:
        metrics['davies_bouldin'] = np.nan
    
    metrics['n_clusters'] = n_clusters
    metrics['n_outliers'] = np.sum(labels == -1)
    
    # Cluster size distribution
    unique, counts = np.unique(labels_filtered, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
    
    return metrics


# ================================================
# STAGE 1: PCA EXPLORATION
# ================================================
def stage1_pca_exploration(args):
    """Stage 1: Run PCA, generate plots."""
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("\n" + "="*60)
    print("STAGE 1: PCA EXPLORATION")
    print("="*60)

    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded {len(df)} rows from {csv_path}")

    # Feature selection
    default_features = [
        'num_neighbors',
        'mean_neighbor_distance',
        'min_neighbor_distance',
        'free_space_volume'
    ]
    
    if args.interactive:
        feature_columns = select_features_interactive(df, default_features)
    else:
        feature_columns = select_features_from_args(df, args.features, default_features)
    
    if len(feature_columns) < 2:
        print("❌ Error: Need at least 2 features for PCA")
        return None
    
    print(f"\nFeatures selected: {', '.join(feature_columns)}")
    
    # Sanitize data
    df_clean = df.dropna(subset=feature_columns).reset_index(drop=True)
    print(f"✓ Cleaned data: {len(df_clean)} rows (removed {len(df) - len(df_clean)} NaN rows)")

    # Scale features
    X = df_clean[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Standardized features (mean=0, std=1)")

    # PCA
    print("\n" + "-"*60)
    print("Running PCA...")
    print("-"*60)
    
    pca, X_pca = compute_pca(X_scaled, n_components=len(feature_columns))
    
    print(f"\nExplained variance by component:")
    cumsum = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumsum += var
        print(f"  PC{i+1}: {var*100:6.2f}%  (cumulative: {cumsum*100:6.2f}%)")
    
    # Save loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    loadings.to_csv(output_dir / "pca_loadings.csv")
    print(f"\n✓ Saved loadings to: {output_dir / 'pca_loadings.csv'}")

    # Generate scree plot
    plot_scree(pca, output_dir / "pca_scree.pdf")
    print(f"✓ Scree plot generated")

    # Add PCs to dataframe
    for i in range(pca.n_components_):
        df_clean[f"PC{i+1}"] = X_pca[:, i]

    # Save PCA results
    pca_results_path = output_dir / "pca_results.csv"
    df_clean.to_csv(pca_results_path, index=False)
    print(f"✓ Saved PCA results: {pca_results_path}")

    print("\n" + "="*60)
    print("STAGE 1 COMPLETE")
    print("="*60)
    
    return {
        'df': df_clean,
        'pca': pca,
        'X_pca': X_pca,
        'scaler': scaler,
        'feature_columns': feature_columns
    }


# ================================================
# METRICS COMPARISON PLOT
# ================================================
def _plot_metrics_comparison(metrics_df, out_path):
    """
    Bar chart comparing Silhouette, Calinski-Harabasz, and Davies-Bouldin
    across all clustering methods. Each metric gets its own subplot.
    Methods are coloured consistently using tab10.
    """
    methods = metrics_df['Method'].tolist()
    palette = sns.color_palette("tab10", n_colors=len(methods))
    method_colors = {m: palette[i] for i, m in enumerate(methods)}

    metrics_config = [
        ('Silhouette',        'Silhouette Score',        'higher is better',  False),
        ('Calinski_Harabasz', 'Calinski-Harabasz Score', 'higher is better',  False),
        ('Davies_Bouldin',    'Davies-Bouldin Score',    'lower is better',   True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Clustering Methods — Metric Comparison', fontsize=14, fontweight='bold', y=1.02)

    for ax, (col, label, note, invert) in zip(axes, metrics_config):
        values = metrics_df[col].tolist()
        colors = [method_colors[m] for m in methods]

        bars = ax.bar(methods, values, color=colors, alpha=0.82, edgecolor='white', linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max([v for v in values if not np.isnan(v)], default=0) * 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold'
                )

        # Highlight best bar with a border
        valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
        if valid:
            best_idx = min(valid, key=lambda x: x[1])[0] if invert else max(valid, key=lambda x: x[1])[0]
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2.5)

        ax.set_title(f'{label}\n({note})', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel('Method', fontsize=10)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add "best" annotation
        if valid:
            ax.annotate('★ best', xy=(best_idx, values[best_idx]),
                        xytext=(0, 14), textcoords='offset points',
                        ha='center', fontsize=8, color='black',
                        arrowprops=dict(arrowstyle='->', color='black', lw=1))

    plt.tight_layout()
    save_figure(fig, out_path)


# ================================================
# STAGE 2: COMPARE CLUSTERING METHODS
# ================================================
def stage2_compare_methods(stage1_results, args):
    """Stage 2: Compare multiple clustering methods."""
    
    print("\n" + "="*60)
    print("STAGE 2: CLUSTERING METHOD COMPARISON")
    print("="*60)
    
    df = stage1_results['df']
    pca = stage1_results['pca']
    X_pca = stage1_results['X_pca']
    output_dir = Path(args.output_dir)
    comparison_dir = output_dir / "comparison"
    ensure_dir(comparison_dir)
    
    # Select PCs for clustering
    if args.cluster_pcs:
        selected_pcs = [int(x)-1 for x in args.cluster_pcs.split(',')]
    else:
        selected_pcs = [0, 1]  # Default PC1, PC2
    
    selected_pcs = [pc for pc in selected_pcs if 0 <= pc < pca.n_components_]
    print(f"\n✓ Using PCs for clustering: {[f'PC{pc+1}' for pc in selected_pcs]}")
    
    X_cluster = X_pca[:, selected_pcs]
    n_clusters = args.n_clusters
    
    # ======================================
    # RUN ALL CLUSTERING METHODS
    # ======================================
    print("\n" + "-"*60)
    print("Running clustering methods...")
    print("-"*60)
    
    results = {}
    
    # 1. KMeans
    print("\n1. KMeans...")
    labels, info = cluster_kmeans(X_cluster, n_clusters, args.random_seed)
    labels = reassign_labels_by_position(X_pca, selected_pcs, labels, n_clusters)
    metrics = compute_metrics(X_cluster, labels)
    results['kmeans'] = {
        'labels': labels,
        'info': info,
        'metrics': metrics,
        'name': 'KMeans'
    }
    print(f"   ✓ {metrics.get('n_clusters', 0)} clusters, Silhouette: {metrics.get('silhouette', np.nan):.3f}")
    
    # 2. GMM
    print("\n2. Gaussian Mixture Model...")
    labels, info = cluster_gmm(X_cluster, n_clusters, args.random_seed)
    labels = reassign_labels_by_position(X_pca, selected_pcs, labels, n_clusters)
    metrics = compute_metrics(X_cluster, labels)
    results['gmm'] = {
        'labels': labels,
        'info': info,
        'metrics': metrics,
        'name': 'GMM'
    }
    print(f"   ✓ {metrics.get('n_clusters', 0)} clusters, Silhouette: {metrics.get('silhouette', np.nan):.3f}")
    
    # 3. DBSCAN
    print("\n3. DBSCAN...")
    labels, info = cluster_dbscan(X_cluster, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    metrics = compute_metrics(X_cluster, labels)
    results['dbscan'] = {
        'labels': labels,
        'info': info,
        'metrics': metrics,
        'name': 'DBSCAN'
    }
    print(f"   ✓ {metrics.get('n_clusters', 0)} clusters, {metrics.get('n_outliers', 0)} outliers, Silhouette: {metrics.get('silhouette', np.nan):.3f}")
    
    # 4. HDBSCAN
    if HDBSCAN_AVAILABLE:
        print("\n4. HDBSCAN...")
        labels, info = cluster_hdbscan(X_cluster, min_cluster_size=args.hdbscan_min_cluster_size, 
                                      min_samples=args.hdbscan_min_samples)
        metrics = compute_metrics(X_cluster, labels)
        results['hdbscan'] = {
            'labels': labels,
            'info': info,
            'metrics': metrics,
            'name': 'HDBSCAN'
        }
        print(f"   ✓ {metrics.get('n_clusters', 0)} clusters, {metrics.get('n_outliers', 0)} outliers, Silhouette: {metrics.get('silhouette', np.nan):.3f}")
    else:
        print("\n4. HDBSCAN... SKIPPED (not installed)")
    
    # 5. Hierarchical
    print("\n5. Hierarchical (Ward)...")
    labels, info = cluster_hierarchical(X_cluster, n_clusters, linkage='ward')
    labels = reassign_labels_by_position(X_pca, selected_pcs, labels, n_clusters)
    metrics = compute_metrics(X_cluster, labels)
    results['hierarchical'] = {
        'labels': labels,
        'info': info,
        'metrics': metrics,
        'name': 'Hierarchical'
    }
    print(f"   ✓ {metrics.get('n_clusters', 0)} clusters, Silhouette: {metrics.get('silhouette', np.nan):.3f}")
    
    # ======================================
    # GENERATE COMPARISON VISUALIZATIONS
    # ======================================
    print("\n" + "-"*60)
    print("Generating comparison visualizations...")
    print("-"*60)
    
    # Individual scatter plots
    for method_key, result in results.items():
        scatter_pca(
            X_pca, selected_pcs[0], selected_pcs[1], 
            result['labels'],
            f"{result['name']} Clustering (PC{selected_pcs[0]+1} vs PC{selected_pcs[1]+1})",
            comparison_dir / f"{method_key}_scatter.pdf",
            pca,
            result['name']
        )
        print(f"   ✓ {result['name']} scatter plot")
    
    # Combined comparison plot
    n_methods = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (method_key, result) in enumerate(results.items()):
        ax = axes[idx]
        labels = result['labels']
        
        unique_labels = np.unique(labels)
        n_clusters_plot = len(unique_labels[unique_labels >= 0])
        palette = sns.color_palette("tab10", n_colors=max(10, n_clusters_plot))
        
        for label in unique_labels:
            if label == -1:
                mask = labels == label
                ax.scatter(X_pca[mask, selected_pcs[0]], X_pca[mask, selected_pcs[1]], 
                          s=20, alpha=0.4, c='black', marker='x')
            else:
                mask = labels == label
                ax.scatter(X_pca[mask, selected_pcs[0]], X_pca[mask, selected_pcs[1]], 
                          s=30, alpha=0.7, color=palette[label % len(palette)])
        
        sil = result['metrics'].get('silhouette', np.nan)
        n_clust = result['metrics'].get('n_clusters', 0)
        ax.set_title(f"{result['name']}\n{n_clust} clusters, Sil: {sil:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel(f"PC{selected_pcs[0]+1}", fontsize=10)
        ax.set_ylabel(f"PC{selected_pcs[1]+1}", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_methods, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_figure(fig, comparison_dir / "all_methods_comparison.pdf")
    print(f"   ✓ Combined comparison plot")
    
    # ======================================
    # METRICS COMPARISON TABLE
    # ======================================
    print("\n" + "-"*60)
    print("Clustering Metrics Comparison")
    print("-"*60)
    
    metrics_data = []
    for method_key, result in results.items():
        m = result['metrics']
        metrics_data.append({
            'Method': result['name'],
            'N_Clusters': m.get('n_clusters', 0),
            'N_Outliers': m.get('n_outliers', 0),
            'Silhouette': m.get('silhouette', np.nan),
            'Calinski_Harabasz': m.get('calinski_harabasz', np.nan),
            'Davies_Bouldin': m.get('davies_bouldin', np.nan),
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('Silhouette', ascending=False)
    
    print("\n" + metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_df.to_csv(comparison_dir / "metrics_comparison.csv", index=False)
    print(f"\n✓ Metrics saved to: {comparison_dir / 'metrics_comparison.csv'}")

    # ======================================
    # METRICS COMPARISON BAR CHART
    # ======================================
    _plot_metrics_comparison(metrics_df, comparison_dir / "metrics_comparison.pdf")
    print(f"   ✓ Metrics comparison bar chart")
    
    # ======================================
    # RECOMMENDATIONS
    # ======================================
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Find best method by silhouette score
    valid_methods = metrics_df[~metrics_df['Silhouette'].isna()]
    if len(valid_methods) > 0:
        best_method = valid_methods.iloc[0]['Method']
        best_sil = valid_methods.iloc[0]['Silhouette']
        print(f"\n🏆 Best method by Silhouette score: {best_method} ({best_sil:.3f})")
        
        if best_sil > 0.7:
            print("   → Excellent clustering quality!")
        elif best_sil > 0.5:
            print("   → Good clustering quality")
        elif best_sil > 0.3:
            print("   → Moderate clustering quality")
        else:
            print("   → Weak clustering - consider different features or preprocessing")
    
    print("\n📊 Interpretation Guide:")
    print("   • Silhouette Score: -1 to 1 (higher is better, >0.5 is good)")
    print("   • Calinski-Harabasz: >0 (higher is better, measures separation)")
    print("   • Davies-Bouldin: >0 (LOWER is better, measures compactness)")
    
    print("\n" + "="*60)
    print("STAGE 2 COMPLETE")
    print("="*60)
    
    return results, metrics_df


# ================================================
# STAGE 3: CNN DATASET WITH SELECTED METHOD
# ================================================
def load_and_compress_image(path, size):
    """Load → grayscale → resize → flatten."""
    try:
        img = Image.open(path).convert("L")
        img = img.resize((size, size))
        return np.array(img).flatten()
    except:
        return None


def stage3_generate_cnn_dataset(stage1_results, clustering_results, args):
    """Stage 3: Generate CNN dataset with selected clustering method."""
    
    print("\n" + "="*60)
    print("STAGE 3: CNN DATASET GENERATION")
    print("="*60)
    
    df = stage1_results['df']
    output_dir = Path(args.output_dir)
    
    # Select best method
    if args.selected_method and args.selected_method in clustering_results:
        method_key = args.selected_method
        print(f"\n✓ Using specified method: {clustering_results[method_key]['name']}")
    else:
        # Auto-select best by silhouette
        best_method = None
        best_score = -999
        for key, result in clustering_results.items():
            sil = result['metrics'].get('silhouette', -999)
            if sil > best_score:
                best_score = sil
                best_method = key
        method_key = best_method
        print(f"\n✓ Auto-selected best method: {clustering_results[method_key]['name']} (Silhouette: {best_score:.3f})")
    
    labels = clustering_results[method_key]['labels']
    df['cluster'] = labels
    
    # Filter out outliers if present
    n_outliers = np.sum(labels == -1)
    if n_outliers > 0:
        print(f"\n⚠ Removing {n_outliers} outliers from CNN dataset")
        df = df[df['cluster'] >= 0].reset_index(drop=True)
    
    print(f"\nCluster distribution:")
    cluster_counts = df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # ======================================
    # CNN DATASET GENERATION
    # ======================================
    print("\n" + "-"*60)
    print("Generating compressed grayscale CNN dataset...")
    print("-"*60)
    print(f"Image size: {args.img_size}x{args.img_size} pixels")
    print("This may take a while for large datasets...\n")

    cnn_rows = []
    img_size = args.img_size
    image_column = args.image_column
    
    failed_count = 0
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images...")
        
        img_path = row[image_column]
        if not isinstance(img_path, str) or not Path(img_path).exists():
            failed_count += 1
            continue

        compressed = load_and_compress_image(img_path, img_size)
        if compressed is None:
            failed_count += 1
            continue

        cnn_rows.append([img_path, row["cluster"], *compressed])

    cnn_df = pd.DataFrame(
        cnn_rows,
        columns=["image_path", "cluster"] + [f"px_{i}" for i in range(img_size*img_size)]
    )

    cnn_output_path = output_dir / f"cnn_dataset_{method_key}.csv"
    cnn_df.to_csv(cnn_output_path, index=False)

    print(f"\n✓ CNN dataset saved: {cnn_output_path}")
    print(f"  Method used: {clustering_results[method_key]['name']}")
    print(f"  Total images processed: {len(cnn_df)}")
    print(f"  Failed/missing images: {failed_count}")
    print(f"  Dataset shape: {cnn_df.shape}")
    
    # Save final clustered results
    clustered_results_path = output_dir / f"clustered_results_{method_key}.csv"
    df.to_csv(clustered_results_path, index=False)
    print(f"\n✓ Final results with clusters saved: {clustered_results_path}")
    
    print("\n" + "="*60)
    print("STAGE 3 COMPLETE - All done!")
    print("="*60)


# ================================================
# MAIN PIPELINE
# ================================================
def main(args):
    stage1_results = stage1_pca_exploration(args)
    if stage1_results is None:
        return
    
    if args.skip_clustering:
        print("\n✓ Skipping clustering (--skip_clustering flag)")
        return
    
    if args.interactive:
        print("\n" + "="*60)
        proceed = input("Proceed to clustering comparison? (y/n): ").strip().lower()
        if proceed != 'y':
            print("✓ Stopped at Stage 1.")
            return
    
    clustering_results, metrics_df = stage2_compare_methods(stage1_results, args)
    
    if args.skip_cnn:
        print("\n✓ Skipping CNN dataset generation (--skip_cnn flag)")
        return
    
    if args.interactive:
        print("\n" + "="*60)
        print("Available methods for CNN dataset:")
        for key, result in clustering_results.items():
            sil = result['metrics'].get('silhouette', np.nan)
            n_clust = result['metrics'].get('n_clusters', 0)
            print(f"  - {key}: {result['name']} ({n_clust} clusters, Sil: {sil:.3f})")
        
        method_choice = input("\nEnter method name for CNN dataset (or ENTER for best): ").strip().lower()
        if method_choice and method_choice in clustering_results:
            args.selected_method = method_choice
        
        proceed = input("Generate CNN dataset? (y/n): ").strip().lower()
        if proceed != 'y':
            print("✓ Stopped after comparison.")
            return
    
    stage3_generate_cnn_dataset(stage1_results, clustering_results, args)


# ================================================
# ARG PARSER
# ================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Method Clustering Comparison Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python method_comparison.py --csv_path data.csv --interactive
  
  # Auto-compare all methods
  python method_comparison.py --csv_path data.csv
  
  # Compare but skip CNN dataset
  python method_comparison.py --csv_path data.csv --skip_cnn
  
  # Use specific method
  python method_comparison.py --csv_path data.csv --selected_method hdbscan
  
  # Custom DBSCAN parameters
  python method_comparison.py --csv_path data.csv --dbscan_eps 0.3 --dbscan_min_samples 10
        """
    )

    parser.add_argument('--csv_path', type=str,
                        default=str(DATA_DIR / 'classifier' / 'mujoco' / 'diverse_local_complexity_data_spatial_features.csv'))
    parser.add_argument('--image_column', type=str, default='image_pixels')
    parser.add_argument('--output_dir', type=str, default=str(DATA_DIR / 'classifier' / 'method_comparison'))
    parser.add_argument('--features', nargs='+')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--cluster_pcs', type=str)
    parser.add_argument('--skip_clustering', action='store_true')
    parser.add_argument('--skip_cnn', action='store_true')
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--dbscan_eps', type=float, default=0.5)
    parser.add_argument('--dbscan_min_samples', type=int, default=5)
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=50)
    parser.add_argument('--hdbscan_min_samples', type=int, default=30)
    parser.add_argument('--selected_method', type=str, default=None,
                       choices=['kmeans', 'gmm', 'dbscan', 'hdbscan', 'hierarchical'])
    parser.add_argument('--img_size', type=int, default=64)

    args = parser.parse_args()
    main(args)