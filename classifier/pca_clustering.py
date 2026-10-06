"""
Two-Stage PCA + KMeans Pipeline with Cluster Reassignment
Stage 1: PCA exploration (no image loading)
Stage 2: KMeans clustering + CNN dataset generation (only after confirmation)
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Repository root (classifier/ -> repo) and git-ignored data directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


# ================================================
# UTIL FUNCTIONS
# ================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_figure(fig, out_path):
    out_path = out_path.with_suffix('.pdf')
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
    
    # Add percentage labels on bars
    for i, v in enumerate(ratios*100):
        ax.text(i+1, v+1, f'{v:.1f}%', ha='center', fontsize=8)
    
    save_figure(fig, out_path)

def plot_loadings_heatmap(loadings_df, out_path):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6*loadings_df.shape[0])))
    sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='vlag', center=0, 
                ax=ax, cbar_kws={'label': 'Loading Weight'})
    ax.set_title('PCA Loadings (Features × Components)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Principal Components', fontsize=11)
    ax.set_ylabel('Features', fontsize=11)
    save_figure(fig, out_path)

def plot_contribution_bars(loadings_df, pc_idx, out_path):
    fig, ax = plt.subplots(figsize=(10,5))
    series = loadings_df.iloc[:, pc_idx].sort_values(key=lambda x: np.abs(x), ascending=False)
    colors = ['green' if x > 0 else 'red' for x in series.values]
    series.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
    ax.set_xlabel('Loading Weight', fontsize=11)
    ax.set_title(f'Feature Contributions to PC{pc_idx+1} ' + 
                 f'({loadings_df.columns[pc_idx]} - explains {loadings_df.iloc[0, pc_idx]:.1f}% variance)',
                 fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    save_figure(fig, out_path)


def scatter_pca(X_pca, pc_x, pc_y, labels=None, title='PCA scatter', out_path=None, pca=None):
    """
    Plot scatter of any two PCs.
    pc_x, pc_y: 0-indexed (e.g., 0 for PC1, 2 for PC3)
    """
    fig, ax = plt.subplots(figsize=(8,7))
    
    variance_x = pca.explained_variance_ratio_[pc_x]*100 if pca else 0
    variance_y = pca.explained_variance_ratio_[pc_y]*100 if pca else 0
    
    if labels is None:
        ax.scatter(X_pca[:,pc_x], X_pca[:,pc_y], s=25, alpha=0.6, c='steelblue')
    else:
        uniq = np.unique(labels)
        palette = sns.color_palette("tab10", n_colors=len(uniq))
        for i, u in enumerate(uniq):
            mask = labels == u
            ax.scatter(X_pca[mask,pc_x], X_pca[mask,pc_y], s=30, alpha=0.7, 
                      label=f'Cluster {u}', color=palette[i])
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    xlabel = f'PC{pc_x+1} ({variance_x:.1f}% variance)'
    ylabel = f'PC{pc_y+1} ({variance_y:.1f}% variance)'
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path)


# ================================================
# CLUSTER REASSIGNMENT
# ================================================
def reassign_clusters_by_position(df, X_pca, selected_pcs, kmeans_labels, n_clusters=3):
    """
    Reassign cluster labels based on spatial position in PC space.
    
    Rules for 3 clusters:
    - Cluster 0: leftmost (lowest PC1)
    - Cluster 1: top-right (high PC1, high PC2)
    - Cluster 2: bottom-right (high PC1, low PC2)
    
    Args:
        df: DataFrame with original data
        X_pca: PCA-transformed data (all components)
        selected_pcs: List of PC indices used for clustering (e.g., [0, 1] for PC1, PC2)
        kmeans_labels: Original cluster labels from KMeans
        n_clusters: Number of clusters
    
    Returns:
        new_labels: Reassigned cluster labels
    """
    # Calculate centroids for each original cluster
    centroids = []
    for k in range(n_clusters):
        mask = kmeans_labels == k
        # Use the PCs that were used for clustering
        centroid = X_pca[mask][:, selected_pcs].mean(axis=0)
        centroids.append((k, centroid))
    
    # Sort centroids by position to determine new labels
    # Assuming selected_pcs contains at least PC1 (index 0) and PC2 (index 1)
    pc1_idx = 0  # Index within selected_pcs for PC1
    pc2_idx = 1 if len(selected_pcs) > 1 else 0  # Index within selected_pcs for PC2
    
    # Create mapping: old_label -> new_label
    label_mapping = {}
    
    if n_clusters == 3:
        # Find leftmost cluster (lowest PC1)
        leftmost = min(centroids, key=lambda x: x[1][pc1_idx])
        label_mapping[leftmost[0]] = 0
        
        # Among remaining two clusters, separate by PC2
        remaining = [c for c in centroids if c[0] != leftmost[0]]
        if len(selected_pcs) > 1:
            # Sort remaining by PC2: higher PC2 = top-right (cluster 1), lower PC2 = bottom-right (cluster 2)
            remaining_sorted = sorted(remaining, key=lambda x: x[1][pc2_idx], reverse=True)
            label_mapping[remaining_sorted[0][0]] = 1  # Higher PC2 -> cluster 1 (top-right)
            label_mapping[remaining_sorted[1][0]] = 2  # Lower PC2 -> cluster 2 (bottom-right)
        else:
            # If only using PC1, rightmost is the one with highest PC1
            top_right = max(remaining, key=lambda x: x[1][pc1_idx])
            label_mapping[top_right[0]] = 1
            last = [c for c in remaining if c[0] != top_right[0]][0]
            label_mapping[last[0]] = 2
        
    else:
        # For other numbers of clusters, simple left-to-right assignment
        sorted_centroids = sorted(centroids, key=lambda x: x[1][pc1_idx])
        for new_label, (old_label, _) in enumerate(sorted_centroids):
            label_mapping[old_label] = new_label
    
    # Apply mapping
    new_labels = np.array([label_mapping[old] for old in kmeans_labels])
    
    # Print mapping for verification
    print("\nCluster label reassignment:")
    for old_label, new_label in sorted(label_mapping.items()):
        old_count = np.sum(kmeans_labels == old_label)
        old_centroid = [centroids[old_label][1][i] for i in range(len(selected_pcs))]
        pc_str = ", ".join([f"PC{selected_pcs[i]+1}={old_centroid[i]:.3f}" 
                           for i in range(len(selected_pcs))])
        print(f"  Old cluster {old_label} ({old_count} samples, centroid: {pc_str}) → New cluster {new_label}")
    
    return new_labels


# ================================================
# STAGE 1: PCA EXPLORATION
# ================================================
def stage1_pca_exploration(args):
    """Stage 1: Run PCA, generate plots, explore without loading images."""
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("\n" + "="*60)
    print("STAGE 1: PCA EXPLORATION (No Image Loading)")
    print("="*60)

    # Load CSV
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

    # ======================================
    # PCA
    # ======================================
    print("\n" + "-"*60)
    print("Running PCA...")
    print("-"*60)
    
    pca, X_pca = compute_pca(X_scaled, n_components=len(feature_columns))
    
    # Print variance explained
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
    
    # Add variance to loadings for reference
    variance_row = pd.DataFrame(
        [pca.explained_variance_ratio_ * 100],
        index=['Variance %'],
        columns=loadings.columns
    )
    loadings_with_var = pd.concat([variance_row, loadings])
    loadings_with_var.to_csv(output_dir / "pca_loadings.csv")
    print(f"\n✓ Saved loadings to: {output_dir / 'pca_loadings.csv'}")

    # Generate plots
    print("\nGenerating PCA plots...")
    plot_scree(pca, output_dir / "pca_scree.png")
    print(f"  ✓ Scree plot")
    
    plot_loadings_heatmap(loadings, output_dir / "pca_loadings_heatmap.png")
    print(f"  ✓ Loadings heatmap")

    # Contribution bars for top components
    n_contribution_plots = min(args.n_contribution_plots, pca.n_components_)
    for i in range(n_contribution_plots):
        plot_contribution_bars(loadings, i, output_dir / f'pca_pc{i+1}_contributions.png')
        print(f"  ✓ PC{i+1} contribution bars")

    # Add PCs to dataframe
    for i in range(pca.n_components_):
        df_clean[f"PC{i+1}"] = X_pca[:, i]

    # Generate scatter plots for specified PC pairs
    print("\nGenerating PC scatter plots...")
    
    if args.pc_pairs:
        # User-specified pairs
        for pair_str in args.pc_pairs:
            pc_x, pc_y = map(int, pair_str.split(','))
            pc_x -= 1  # Convert to 0-indexed
            pc_y -= 1
            
            if pc_x >= pca.n_components_ or pc_y >= pca.n_components_:
                print(f"  ⚠ Skipping PC{pc_x+1} vs PC{pc_y+1} (not enough components)")
                continue
            
            scatter_pca(X_pca, pc_x, pc_y, None, 
                       f"PCA: PC{pc_x+1} vs PC{pc_y+1}", 
                       output_dir / f"pca_scatter_PC{pc_x+1}_vs_PC{pc_y+1}.png",
                       pca)
            print(f"  ✓ PC{pc_x+1} vs PC{pc_y+1}")
    else:
        # Default: plot first 3 PCs in combinations
        max_pc = min(3, pca.n_components_)
        for pc_x, pc_y in combinations(range(max_pc), 2):
            scatter_pca(X_pca, pc_x, pc_y, None, 
                       f"PCA: PC{pc_x+1} vs PC{pc_y+1}", 
                       output_dir / f"pca_scatter_PC{pc_x+1}_vs_PC{pc_y+1}.png",
                       pca)
            print(f"  ✓ PC{pc_x+1} vs PC{pc_y+1}")

    # Save PCA results
    pca_results_path = output_dir / "pca_results.csv"
    df_clean.to_csv(pca_results_path, index=False)
    print(f"\n✓ Saved PCA results with all PCs to: {pca_results_path}")

    print("\n" + "="*60)
    print("STAGE 1 COMPLETE - Review the plots and decide on clustering")
    print("="*60)
    
    return {
        'df': df_clean,
        'pca': pca,
        'X_pca': X_pca,
        'scaler': scaler,
        'feature_columns': feature_columns
    }


# ================================================
# STAGE 2: CLUSTERING + IMAGE PROCESSING
# ================================================
def load_and_compress_image(path, size):
    """Load → grayscale → resize → flatten."""
    try:
        img = Image.open(path).convert("L")
        img = img.resize((size, size))
        return np.array(img).flatten()
    except:
        return None


def stage2_clustering_and_images(stage1_results, args):
    """Stage 2: Apply KMeans on selected PCs and generate CNN dataset."""
    
    print("\n" + "="*60)
    print("STAGE 2: CLUSTERING + CNN DATASET GENERATION")
    print("="*60)
    
    df = stage1_results['df']
    pca = stage1_results['pca']
    X_pca = stage1_results['X_pca']
    output_dir = Path(args.output_dir)
    csv_dir = Path(args.csv_path).resolve().parent
    
    # Select PCs for clustering
    print(f"\nAvailable PCs: {pca.n_components_}")
    print("Select which PCs to use for clustering:")
    
    if args.cluster_pcs:
        selected_pcs = [int(x)-1 for x in args.cluster_pcs.split(',')]
    else:
        if args.interactive:
            pc_input = input(f"Enter PC numbers separated by commas (default: 1,2): ").strip()
            if not pc_input:
                selected_pcs = [0, 1]
            else:
                selected_pcs = [int(x.strip())-1 for x in pc_input.split(',')]
        else:
            selected_pcs = [0, 1]  # Default PC1, PC2
    
    selected_pcs = [pc for pc in selected_pcs if 0 <= pc < pca.n_components_]
    print(f"✓ Using PCs for clustering: {[f'PC{pc+1}' for pc in selected_pcs]}")
    
    X_cluster = X_pca[:, selected_pcs]
    
    # Get number of clusters
    if args.interactive and not args.kmeans_n_clusters:
        n_clusters_input = input(f"\nNumber of clusters (default: 3): ").strip()
        n_clusters = int(n_clusters_input) if n_clusters_input else 3
    else:
        n_clusters = args.kmeans_n_clusters
    
    print(f"✓ Using {n_clusters} clusters")
    
    # ======================================
    # KMEANS CLUSTERING WITH REASSIGNMENT
    # ======================================
    print("\n" + "-"*60)
    print(f"Running KMeans clustering on selected PCs...")
    print("-"*60)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.random_seed, n_init=10)
    original_labels = kmeans.fit_predict(X_cluster)
    
    # REASSIGN CLUSTERS BASED ON POSITION
    new_labels = reassign_clusters_by_position(df, X_pca, selected_pcs, original_labels, n_clusters)
    df["kmeans_cluster"] = new_labels
    
    print(f"\nFinal cluster distribution:")
    cluster_counts = df["kmeans_cluster"].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Cluster metrics
    try:
        sil = silhouette_score(X_cluster, df["kmeans_cluster"])
        ch = calinski_harabasz_score(X_cluster, df["kmeans_cluster"])
        print(f"\nClustering metrics:")
        print(f"  Silhouette score: {sil:.4f} (higher is better, range: -1 to 1)")
        print(f"  Calinski-Harabasz: {ch:.2f} (higher is better)")
    except:
        sil, ch = np.nan, np.nan
        print(f"\n⚠ Could not compute clustering metrics")

    pd.DataFrame({
        "n_clusters": [n_clusters],
        "pcs_used": [','.join([f'PC{pc+1}' for pc in selected_pcs])],
        "silhouette_score": [sil],
        "calinski_harabasz": [ch],
    }).to_csv(output_dir / "kmeans_metrics.csv", index=False)
    
    # Generate cluster scatter plots
    print("\nGenerating cluster scatter plots...")
    if len(selected_pcs) >= 2:
        scatter_pca(X_pca, selected_pcs[0], selected_pcs[1], 
                   df["kmeans_cluster"], 
                   f"KMeans {n_clusters} Clusters (PC{selected_pcs[0]+1} vs PC{selected_pcs[1]+1})", 
                   output_dir / f"kmeans_scatter_PC{selected_pcs[0]+1}_vs_PC{selected_pcs[1]+1}.png",
                   pca)
        print(f"  ✓ PC{selected_pcs[0]+1} vs PC{selected_pcs[1]+1} with clusters")
    
    # Also plot on PC1 vs PC2 if different from selected PCs
    if selected_pcs != [0, 1] and pca.n_components_ >= 2:
        scatter_pca(X_pca, 0, 1, 
                   df["kmeans_cluster"], 
                   f"KMeans {n_clusters} Clusters (PC1 vs PC2)", 
                   output_dir / f"kmeans_scatter_PC1_vs_PC2.png",
                   pca)
        print(f"  ✓ PC1 vs PC2 with clusters")

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
        # Older CSVs may store paths relative to the CSV folder instead of absolute paths.
        if isinstance(img_path, str) and not Path(img_path).exists() \
                and (csv_dir / img_path).exists():
            img_path = str((csv_dir / img_path).resolve())
            df.at[idx, image_column] = img_path

        if not isinstance(img_path, str) or not Path(img_path).exists():
            failed_count += 1
            continue

        compressed = load_and_compress_image(img_path, img_size)
        if compressed is None:
            failed_count += 1
            continue

        cnn_rows.append(
            [img_path, row["kmeans_cluster"], *compressed]
        )

    cnn_df = pd.DataFrame(
        cnn_rows,
        columns=["image_path", "cluster"] + [f"px_{i}" for i in range(img_size*img_size)]
    )

    cnn_output_path = output_dir / "cnn_dataset.csv"
    cnn_df.to_csv(cnn_output_path, index=False)

    print(f"\n✓ CNN dataset saved: {cnn_output_path}")
    print(f"  Total images processed: {len(cnn_df)}")
    print(f"  Failed/missing images: {failed_count}")
    print(f"  Dataset shape: {cnn_df.shape}")
    
    # Save final clustered results
    clustered_results_path = output_dir / "pca_kmeans_results.csv"
    df.to_csv(clustered_results_path, index=False)
    print(f"\n✓ Final results with clusters saved: {clustered_results_path}")
    
    print("\n" + "="*60)
    print("STAGE 2 COMPLETE - All done!")
    print("="*60)


# ================================================
# MAIN PIPELINE
# ================================================
def main(args):
    # Stage 1: PCA exploration (no images)
    stage1_results = stage1_pca_exploration(args)
    
    if stage1_results is None:
        return
    
    # Decide whether to continue to Stage 2
    if args.skip_clustering:
        print("\n✓ Skipping clustering and image processing (--skip_clustering flag)")
        return
    
    if args.interactive:
        print("\n" + "="*60)
        proceed = input("Proceed to clustering and image processing? (y/n): ").strip().lower()
        if proceed != 'y':
            print("✓ Stopped at Stage 1. Review the plots and run Stage 2 later if needed.")
            return
    
    # Stage 2: Clustering and image processing
    stage2_clustering_and_images(stage1_results, args)


# ================================================
# ARG PARSER
# ================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Stage PCA + KMeans Pipeline with Cluster Reassignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (choose features, confirm before clustering)
  python pca_clustering.py --csv_path data.csv --interactive
  
  # Explore specific features and PC pairs without clustering
  python pca_clustering.py --csv_path data.csv --features target_grasp_difficulty num_neighbors --pc_pairs 1,2 1,3 2,3 --skip_clustering
  
  # Full pipeline with specific settings
  python pca_clustering.py --csv_path data.csv --features target_grasp_difficulty target_shape_complexity num_neighbors --cluster_pcs 1,3 --kmeans_n_clusters 4
        """
    )

    # Input/Output
    parser.add_argument('--csv_path', type=str,
                       default=str(DATA_DIR / 'classifier' / 'mujoco' / 'diverse_local_complexity_data_spatial_features.csv'),
                       help='Path to input CSV file (MuJoCo: output of spatial_features_extraction_raw.py; '
                            'Isaac: classifier_data.csv from isaac/collect_classifier_dataset.py)')
    parser.add_argument('--image_column', type=str, default='image_pixels',
                       help='Column name containing image paths')
    parser.add_argument('--output_dir', type=str, default=str(DATA_DIR / 'classifier' / 'pca_kmeans'),
                       help='Output directory for results')

    # Feature selection
    parser.add_argument('--features', nargs='+', 
                       help='Feature columns to use (space-separated)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive feature selection and confirmation')

    # PCA visualization
    parser.add_argument('--pc_pairs', nargs='+',
                       help='PC pairs to plot, e.g., --pc_pairs 1,2 1,3 2,4')
    parser.add_argument('--n_contribution_plots', type=int, default=4,
                       help='Number of PC contribution bar plots to generate')

    # Clustering
    parser.add_argument('--skip_clustering', action='store_true',
                       help='Skip Stage 2 (clustering and image processing)')
    parser.add_argument('--cluster_pcs', type=str,
                       help='PCs to use for clustering (comma-separated, e.g., 1,3)')
    parser.add_argument('--kmeans_n_clusters', type=int, default=3,
                       help='Number of clusters for KMeans')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Image processing
    parser.add_argument('--img_size', type=int, default=64,
                       help='Compressed grayscale image size (img_size x img_size)')

    args = parser.parse_args()
    main(args)