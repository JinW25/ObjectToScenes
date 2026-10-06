"""
Extract bbox and 3D position features from existing rendered images.
No need to regenerate - we can parse from the images you already have!
"""

import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# Repository root (classifier/mujoco/ -> repo) and git-ignored data directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def resolve_image_path(img_path, csv_dir):
    """Return img_path as-is if it exists; otherwise try it relative to the CSV's folder.

    New CSVs store absolute paths; this keeps older CSVs with CWD-relative paths usable.
    """
    if not isinstance(img_path, str):
        return img_path
    if os.path.isabs(img_path) or os.path.exists(img_path):
        return img_path
    candidate = os.path.join(csv_dir, img_path)
    return candidate if os.path.exists(candidate) else img_path

def detect_green_bbox_from_image(image_path):
    """
    Detect the green bounding box from your rendered images.
    Your images have a lime/green box drawn on them - we can extract it!
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = img_hsv.shape[:2]
        
        # Create mask for green color (lime color you used)
        # Lime is approximately (0, 255, 0) in RGB
        lower_green = np.array([55, 240, 240])
        upper_green = np.array([65, 255, 255])
        
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest rectangular contour (should be your bbox)
        max_area = 0
        best_bbox = None
        
        for contour in contours:
            # Approximate to rectangle
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter out noise (too small or wrong aspect ratio)
            if area > 100 and 0.1 < w/h < 10:
                if area > max_area:
                    max_area = area
                    best_bbox = (x, y, x + w, y + h)
        
        if best_bbox is None:
            return None
        
        x_min, y_min, x_max, y_max = best_bbox
        
        # Normalize to 0-1
        bbox_features = {
            'bbox_x_min': x_min / width,
            'bbox_x_max': x_max / width,
            'bbox_y_min': y_min / height,
            'bbox_y_max': y_max / height,
            'bbox_center_x': ((x_min + x_max) / 2) / width,
            'bbox_center_y': ((y_min + y_max) / 2) / height,
            'bbox_width': (x_max - x_min) / width,
            'bbox_height': (y_max - y_min) / height,
            'bbox_area': ((x_max - x_min) * (y_max - y_min)) / (width * height),
        }
        
        return bbox_features
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def estimate_3d_position_from_bbox(bbox_features, scene_info):
    """
    Estimate 3D position from bbox and scene context.
    
    3D Coordinate System (Normalized 0-1):
    ----------------------------------------
    Origin: Bottom-left-front corner of workspace
    
    X-axis (0-1): Left to Right
        - 0.0 = Left edge of workspace (x_min = 0.2m)
        - 0.5 = Center
        - 1.0 = Right edge of workspace (x_max = 0.8m)
    
    Y-axis (0-1): Front to Back (depth from camera)
        - 0.0 = Front edge (y_min = -0.45m, closest to camera)
        - 0.5 = Middle
        - 1.0 = Back edge (y_max = 0.45m, furthest from camera)
    
    Z-axis (0-1): Bottom to Top (height)
        - 0.0 = Table surface (z = 1.0m)
        - 0.5 = Mid-height
        - 1.0 = Maximum reach height (z = 1.5m)
    
    Note: These are ESTIMATES based on visual cues:
    - X: Directly from bbox horizontal center
    - Y: From bbox vertical position (top = far, bottom = near)
    - Z: From bbox area (large = close/tall, small = far/short)
    """
    
    # Extract bbox info
    center_x = bbox_features['bbox_center_x']
    center_y = bbox_features['bbox_center_y']
    bbox_area = bbox_features['bbox_area']
    bbox_width = bbox_features['bbox_width']
    bbox_height = bbox_features['bbox_height']
    
    # === X Position (Left-Right) ===
    # Direct mapping from image horizontal center
    obj_x_norm = center_x
    
    # === Y Position (Front-Back / Depth) ===
    # Perspective: objects higher in image are further away
    # Invert Y: top of image (low center_y) = far (high Y)
    obj_y_norm = 0.5 + (0.5 - center_y) * 0.8  # Scale factor for perspective
    
    # === Z Position (Height) ===
    # Estimate from bbox size: larger bbox = closer OR taller
    # We use area as proxy for "how prominent the object is"
    # Typical object area: 0.01-0.05
    # Height estimation: larger area suggests taller/closer object
    base_z = min(1.0, bbox_area / 0.04)  # Normalize by typical max area
    
    # Refine Z using aspect ratio
    # Tall objects (height > width) are likely standing upright
    aspect_ratio = bbox_height / (bbox_width + 1e-6)
    if aspect_ratio > 1.2:  # Tall object
        base_z = min(1.0, base_z * 1.2)  # Boost height estimate
    
    obj_z_norm = base_z
    
    # === Distance to Workspace Center ===
    # Center of workspace is at (0.5, 0.5) in normalized coords
    distance_to_center = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
    distance_to_center_norm = min(1.0, distance_to_center / 0.707)  # Max dist = sqrt(0.5^2 + 0.5^2)
    
    # === Distance to Camera ===
    # Estimate from bbox area: larger area = closer to camera
    # Inverse relationship: large area = close = small distance value
    distance_to_camera_norm = max(0.1, min(1.0, 1.0 - (bbox_area / 0.08)))
    
    # === Refine with Scene Context ===
    if scene_info is not None:
        # If object has many neighbors, it's likely in the middle (dense area)
        num_neighbors = scene_info.get('num_neighbors', 0)
        if num_neighbors > 3:
            # Cluttered center - pull estimates towards center
            obj_x_norm = 0.7 * obj_x_norm + 0.3 * 0.5
            obj_y_norm = 0.7 * obj_y_norm + 0.3 * 0.5
        
        # If free space is low, object is likely surrounded (center)
        free_space = scene_info.get('free_space_volume', 0.5)
        if free_space < 0.3:
            distance_to_center_norm *= 0.8  # Reduce distance to center
        
        # If neighbors are close, object is in dense region (likely table surface)
        mean_neighbor_dist = scene_info.get('mean_neighbor_distance', 1.0)
        if mean_neighbor_dist < 0.05:
            obj_z_norm = min(obj_z_norm, 0.4)  # Likely on table, not elevated
    
    position_features = {
        'obj_x': np.clip(obj_x_norm, 0, 1),
        'obj_y': np.clip(obj_y_norm, 0, 1),
        'obj_z': np.clip(obj_z_norm, 0, 1),
        'distance_to_center': np.clip(distance_to_center_norm, 0, 1),
        'distance_to_camera': np.clip(distance_to_camera_norm, 0, 1),
    }
    
    return position_features


def add_spatial_features_to_csv(csv_path, output_csv_path, assume_yes=False):
    """
    Main function: Add bbox and 3D position features to existing CSV.
    """
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Check if spatial features already exist
    spatial_cols = ['bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height', 'bbox_area',
                    'obj_x', 'obj_y', 'obj_z', 'distance_to_center', 'distance_to_camera']
    
    if all(col in df.columns for col in spatial_cols):
        print("\n Spatial features already exist in CSV!")
        response = 'y' if assume_yes else input("Overwrite existing features? (y/n): ")
        if response.lower() != 'y':
            return df
    
    # Initialize new columns
    for col in spatial_cols:
        df[col] = np.nan
    
    print("\nExtracting spatial features from images...")
    
    success_count = 0
    fail_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = resolve_image_path(row['image_pixels'], csv_dir)
        
        # Extract bbox from image
        bbox_features = detect_green_bbox_from_image(img_path)
        
        if bbox_features is None:
            fail_count += 1
            continue
        
        # Estimate 3D position from bbox + scene context
        scene_info = None
        
        position_features = estimate_3d_position_from_bbox(bbox_features, scene_info)
        
        # Update dataframe
        for key, value in bbox_features.items():
            df.at[idx, key] = value
        
        for key, value in position_features.items():
            df.at[idx, key] = value
        
        success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Success: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
    print(f"  Failed:  {fail_count}/{len(df)} ({fail_count/len(df)*100:.1f}%)")
    print(f"{'='*60}")
    
    # Remove rows where extraction failed
    df_clean = df.dropna(subset=spatial_cols).reset_index(drop=True)
    print(f"\nRows after cleaning: {len(df_clean)}")
    
    # Save updated CSV
    df_clean.to_csv(output_csv_path, index=False)
    print(f"\nSaved updated CSV to {output_csv_path}")
    
    return df_clean


def visualize_extracted_features(df, num_samples=10, output_path='spatial_features_viz_raw.png', csv_dir='.'):
    """
    Visualize extracted features overlaid on images with red bounding boxes.
    Shows both the detected green box and extracted red box with annotations.
    """
    samples = df.sample(min(num_samples, len(df)))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 5, 10))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (_, row) in enumerate(samples.iterrows()):
        img_path = resolve_image_path(row['image_pixels'], csv_dir)
        
        try:
            # Load image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]
            
            # Create a copy for drawing
            img_annotated = img_rgb.copy()
            
            # Extract bbox coordinates (denormalize)
            x_min = int(row['bbox_x_min'] * width)
            y_min = int(row['bbox_y_min'] * height)
            x_max = int(row['bbox_x_max'] * width)
            y_max = int(row['bbox_y_max'] * height)
            center_x = int(row['bbox_center_x'] * width)
            center_y = int(row['bbox_center_y'] * height)
            
            # Draw RED bounding box (extracted)
            cv2.rectangle(img_annotated, (x_min, y_min), (x_max, y_max), 
                         (255, 0, 0), 3)  # Red, thick line
            
            # Draw center point
            cv2.circle(img_annotated, (center_x, center_y), 8, (255, 0, 0), -1)
            
            # Add bbox dimension annotations
            # Width line
            cv2.line(img_annotated, (x_min, y_max + 15), (x_max, y_max + 15), 
                    (255, 255, 0), 2)
            cv2.putText(img_annotated, f"W:{row['bbox_width']:.3f}", 
                       (x_min, y_max + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
            
            # Height line
            cv2.line(img_annotated, (x_max + 15, y_min), (x_max + 15, y_max), 
                    (255, 255, 0), 2)
            cv2.putText(img_annotated, f"H:{row['bbox_height']:.3f}", 
                       (x_max + 20, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
            
            # Add 3D position visualization
            # Create a small coordinate system indicator
            coord_x = 30
            coord_y = height - 100
            
            # Draw coordinate axes
            axis_length = 50
            cv2.arrowedLine(img_annotated, (coord_x, coord_y), 
                          (coord_x + axis_length, coord_y), 
                          (255, 0, 0), 2, tipLength=0.3)  # X-axis (red)
            cv2.putText(img_annotated, "X", (coord_x + axis_length + 5, coord_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.arrowedLine(img_annotated, (coord_x, coord_y), 
                          (coord_x, coord_y - axis_length), 
                          (0, 255, 0), 2, tipLength=0.3)  # Y-axis (green)
            cv2.putText(img_annotated, "Y", (coord_x - 5, coord_y - axis_length - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.arrowedLine(img_annotated, (coord_x, coord_y), 
                          (coord_x + int(axis_length * 0.7), coord_y + int(axis_length * 0.7)), 
                          (0, 0, 255), 2, tipLength=0.3)  # Z-axis (blue)
            cv2.putText(img_annotated, "Z", 
                       (coord_x + int(axis_length * 0.7) + 5, coord_y + int(axis_length * 0.7) + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw object position on coordinate system
            obj_x_px = coord_x + int(row['obj_x'] * axis_length)
            obj_y_px = coord_y - int(row['obj_y'] * axis_length)
            cv2.circle(img_annotated, (obj_x_px, obj_y_px), 5, (255, 255, 0), -1)
            
            # Top row: Annotated image with red box
            axes[0, col].imshow(img_annotated)
            axes[0, col].set_title(f"Extracted BBox (RED)", fontsize=10, fontweight='bold')
            axes[0, col].axis('off')
            
            # Bottom row: Feature details with 3D coordinate visualization
            axes[1, col].axis('off')
            
            # Create text annotation
            feature_text = f"""
BBox Features (2D Image):
  Center: ({row['bbox_center_x']:.3f}, {row['bbox_center_y']:.3f})
  Size: {row['bbox_width']:.3f} × {row['bbox_height']:.3f}
  Area: {row['bbox_area']:.4f}

3D Position (Workspace):
  X: {row['obj_x']:.3f} (Left -> 0.5 -> Right)
  Y: {row['obj_y']:.3f} (Near -> 0.5 -> Far)
  Z: {row['obj_z']:.3f} (Low -> 0.5 -> High)

Spatial Metrics:
  Dist to center: {row['distance_to_center']:.3f}
  Dist to camera: {row['distance_to_camera']:.3f}

Scene Context:
  Neighbors: {row.get('num_neighbors', 'N/A')}
  Free space: {row.get('free_space_volume', 'N/A'):.3f}
            """
            
            axes[1, col].text(0.05, 0.95, feature_text.strip(), 
                            transform=axes[1, col].transAxes,
                            fontsize=8, verticalalignment='top',
                            family='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add 3D coordinate system diagram
            axes[1, col].text(0.05, 0.05, 
                            "3D Coords: X(L-R), Y(N-F), Z(L-H)",
                            transform=axes[1, col].transAxes,
                            fontsize=7, style='italic')
            
        except Exception as e:
            axes[0, col].text(0.5, 0.5, f"Error:\n{str(e)}", 
                            ha='center', va='center', fontsize=8)
            axes[0, col].axis('off')
            axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")
    plt.close()


def create_3d_coordinate_diagram(output_path='3d_coordinate_system.png'):
    """
    Create a detailed 3D coordinate system reference diagram.
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Main title
    fig.suptitle('3D Coordinate System - Workspace Reference', 
                 fontsize=16, fontweight='bold')
    
    # Left: Top-down view
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Draw workspace boundary
    workspace = plt.Rectangle((0, 0), 1, 1, fill=False, 
                              edgecolor='black', linewidth=2)
    ax1.add_patch(workspace)
    
    # Mark center
    ax1.plot(0.5, 0.5, 'r*', markersize=20, label='Workspace Center')
    
    # Draw coordinate axes
    ax1.arrow(0.5, 0.5, 0.35, 0, head_width=0.03, head_length=0.05, 
             fc='red', ec='red', linewidth=2)
    ax1.text(0.9, 0.52, 'X (Left -> Right)', fontsize=10, fontweight='bold', color='red')
    
    ax1.arrow(0.5, 0.5, 0, 0.35, head_width=0.03, head_length=0.05, 
             fc='green', ec='green', linewidth=2)
    ax1.text(0.52, 0.9, 'Y (Near -> Far)', fontsize=10, fontweight='bold', 
            color='green', rotation=90)
    
    # Mark corners
    corners = [
        (0, 0, 'Front-Left\n(0, 0)'),
        (1, 0, 'Front-Right\n(1, 0)'),
        (0, 1, 'Back-Left\n(0, 1)'),
        (1, 1, 'Back-Right\n(1, 1)'),
    ]
    for x, y, label in corners:
        ax1.plot(x, y, 'ko', markersize=8)
        ax1.text(x, y - 0.08, label, ha='center', fontsize=8)
    
    # Add example object positions
    examples = [
        (0.3, 0.3, 'Near-Left\nEasy reach'),
        (0.7, 0.7, 'Far-Right\nHard reach'),
        (0.5, 0.2, 'Front-Center\nBest position'),
    ]
    for x, y, label in examples:
        ax1.plot(x, y, 'bs', markersize=10, alpha=0.6)
        ax1.text(x + 0.05, y + 0.05, label, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('X-axis (normalized 0-1)', fontsize=11)
    ax1.set_ylabel('Y-axis (normalized 0-1)', fontsize=11)
    ax1.set_title('Top-Down View (X-Y Plane)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Right: Side view
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Draw table
    table = plt.Rectangle((0, 0), 1, 0.05, fill=True, 
                          facecolor='brown', edgecolor='black', linewidth=2)
    ax2.add_patch(table)
    ax2.text(0.5, -0.05, 'Table Surface (Z=0)', ha='center', fontsize=9)
    
    # Draw Z-axis
    ax2.arrow(0.1, 0, 0, 0.9, head_width=0.03, head_length=0.05, 
             fc='blue', ec='blue', linewidth=2)
    ax2.text(0.15, 0.95, 'Z (Height)', fontsize=10, fontweight='bold', color='blue')
    
    # Mark height levels
    heights = [
        (0.0, 'Table (0.0)'),
        (0.3, 'Low (0.3)'),
        (0.5, 'Mid (0.5)'),
        (0.8, 'High (0.8)'),
        (1.0, 'Max (1.0)'),
    ]
    for z, label in heights:
        ax2.axhline(y=z, color='gray', linestyle='--', alpha=0.5)
        ax2.text(0.92, z, label, fontsize=8, va='center')
    
    # Add example objects at different heights
    ax2.plot([0.3, 0.5, 0.7], [0.2, 0.5, 0.3], 'bs', markersize=12, alpha=0.6)
    ax2.text(0.3, 0.25, 'Short', ha='center', fontsize=8)
    ax2.text(0.5, 0.55, 'Tall', ha='center', fontsize=8)
    ax2.text(0.7, 0.35, 'Medium', ha='center', fontsize=8)
    
    ax2.set_xlabel('Y-axis (depth, normalized 0-1)', fontsize=11)
    ax2.set_ylabel('Z-axis (height, normalized 0-1)', fontsize=11)
    ax2.set_title('Side View (Y-Z Plane)', fontsize=12, fontweight='bold')
    
    # Add coordinate system explanation
    explanation = """
    Coordinate System Summary:
    ==========================================
    - All coordinates normalized to [0, 1]
    - Origin (0,0,0) = Front-Left-Bottom corner
    - Center (0.5,0.5,0.5) = Workspace center
    
    Physical Mapping (from your config):
    - X: 0.0 -> 0.2m, 1.0 -> 0.8m (width: 0.6m)
    - Y: 0.0 -> -0.45m, 1.0 -> 0.45m (depth: 0.9m)
    - Z: 0.0 -> 1.0m (table), 1.0 -> 1.5m (height: 0.5m)
    """
    
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=9, 
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved 3D coordinate diagram to {output_path}")
    plt.close()


def print_feature_statistics(df):
    """Print statistics of extracted features."""
    print("\n" + "="*60)
    print("SPATIAL FEATURE STATISTICS")
    print("="*60)
    
    bbox_cols = ['bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height', 'bbox_area']
    pos_cols = ['obj_x', 'obj_y', 'obj_z', 'distance_to_center', 'distance_to_camera']
    
    print("\nBBox Features:")
    print(df[bbox_cols].describe())
    
    print("\n3D Position Features:")
    print(df[pos_cols].describe())
    
    print("\n" + "="*60)


# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    default_input = DATA_DIR / "classifier" / "mujoco" / "diverse_local_complexity_data.csv"
    parser = argparse.ArgumentParser(
        description="Add bbox / position features (detected from the green target box) to the "
                    "scene CSV from randomize_scene.py (step 2 of the classifier pipeline).")
    parser.add_argument("--input_csv", type=str, default=str(default_input),
                        help="CSV written by randomize_scene.py")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV (default: <input>_spatial_features.csv next to the input)")
    parser.add_argument("--num_viz", type=int, default=5,
                        help="Number of samples in the visualisation figure (0 to skip)")
    parser.add_argument("--yes", action="store_true",
                        help="Overwrite existing spatial feature columns without asking")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv) if args.output_csv else \
        input_csv.with_name(input_csv.stem + "_spatial_features.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    viz_path = output_csv.with_name(output_csv.stem + "_viz.png")
    
    print("="*60)
    print("SPATIAL FEATURE EXTRACTION")
    print("="*60)
    print(f"Input : {input_csv}")
    print(f"Output: {output_csv}")
    print("="*60)
    
    # Extract features
    df = add_spatial_features_to_csv(str(input_csv), str(output_csv), assume_yes=args.yes)
    
    # Print statistics
    print_feature_statistics(df)
    
    # Visualize samples
    if args.num_viz > 0 and len(df) > 0:
        print("\nGenerating visualization...")
        visualize_extracted_features(df, num_samples=min(args.num_viz, len(df)),
                                     output_path=str(viz_path),
                                     csv_dir=str(input_csv.resolve().parent))
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"Updated CSV saved to: {output_csv}")
    print("Next: python classifier/pca_clustering.py --csv_path " + str(output_csv))
    print("="*60)
