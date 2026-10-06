"""Build the classifier dataset (step 4): stratified 70/20/10 train/val/test split of the
K-Means-labelled scenes, images resized to 224x224, 3 augmented copies per training image."""

import os
import shutil
import random
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

import argparse
from pathlib import Path

import torch

# Repository root (classifier/ -> repo) and git-ignored data directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# ======================
# CONFIG
# ======================
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv_path", type=str, default=str(DATA_DIR / "classifier" / "pca_kmeans" / "cnn_dataset.csv"),
                    help="cnn_dataset.csv written by pca_clustering.py")
parser.add_argument("--output_root", type=str, default=str(DATA_DIR / "classifier" / "cnn_dataset"),
                    help="Output folder for train/val/test (deleted and recreated)")
parser.add_argument("--augmentations_per_image", type=int, default=3,
                    help="Number of augmented versions per training image")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

csv_path = args.csv_path   # CSV file from the previous step
image_column = "image_path"  # Column name written by pca_clustering.py
label_column = "cluster"

output_root = args.output_root
train_dir = os.path.join(output_root, "train")
val_dir = os.path.join(output_root, "val")
test_dir = os.path.join(output_root, "test")

# Split ratios: 70% train, 20% val, 10% test
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)  # torchvision random transforms use the torch RNG

augmentations_per_image = args.augmentations_per_image  # Number of augmented versions per image for training

img_resolution = (224, 224)   # For CNNs (e.g., ResNet, EfficientNet)


# ======================
# LOAD LABELED DATA
# ======================
# Only the path and label are needed (the px_* columns are not used here).
df = pd.read_csv(csv_path, usecols=[image_column, label_column])

# Remove noise cluster (-1) if needed
df = df[df[label_column] != -1]

# Filter out rows where image doesn't exist
print("Checking image paths...")
valid_rows = []
for idx, row in df.iterrows():
    img_path = row[image_column]
    if os.path.exists(img_path):
        valid_rows.append(idx)
    else:
        print(f"Warning: Image not found: {img_path}")

df = df.loc[valid_rows].reset_index(drop=True)
print(f"Valid images found: {len(df)}")

clusters = sorted(df[label_column].unique())
print("Clusters found:", clusters)


# ======================
# BALANCED TRAIN/VAL/TEST SPLIT (70:20:10)
# ======================
train_indices = []
val_indices = []
test_indices = []

for c in clusters:
    cluster_df = df[df[label_column] == c]
    cluster_indices = cluster_df.index.tolist()
    
    # First split: separate test set (10%)
    train_val_idx, test_idx = train_test_split(
        cluster_indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True
    )
    
    # Second split: separate train and validation (70:20 from remaining 90%)
    # val_ratio_adjusted = val_ratio / (train_ratio + val_ratio) ≈ 0.222
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=seed,
        shuffle=True
    )
    
    train_indices.extend(train_idx)
    val_indices.extend(val_idx)
    test_indices.extend(test_idx)

train_df = df.loc[train_indices].reset_index(drop=True)
val_df = df.loc[val_indices].reset_index(drop=True)
test_df = df.loc[test_indices].reset_index(drop=True)

print(f"\n{'='*50}")
print(f"Dataset Split Summary:")
print(f"{'='*50}")
print(f"Training samples  : {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Testing samples   : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
print(f"Total samples     : {len(df)}")
print(f"{'='*50}\n")

# Print distribution per cluster
for c in clusters:
    train_count = len(train_df[train_df[label_column] == c])
    val_count = len(val_df[val_df[label_column] == c])
    test_count = len(test_df[test_df[label_column] == c])
    total_count = train_count + val_count + test_count
    
    print(f"Cluster {c}: Train={train_count}, Val={val_count}, Test={test_count} (Total={total_count})")


# ======================
# DATA AUGMENTATION (Training Only)
# ======================
train_transform = transforms.Compose([
    transforms.Resize(img_resolution),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=img_resolution, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

val_test_transform = transforms.Compose([
    transforms.Resize(img_resolution),
    transforms.CenterCrop(img_resolution),
])


# ======================
# CREATE OUTPUT FOLDERS
# ======================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Clean up existing directory
if os.path.exists(output_root):
    shutil.rmtree(output_root)
    print(f"Removed existing directory: {output_root}\n")

for c in clusters:
    ensure_dir(os.path.join(train_dir, f"cluster_{c}"))
    ensure_dir(os.path.join(val_dir, f"cluster_{c}"))
    ensure_dir(os.path.join(test_dir, f"cluster_{c}"))


# ======================
# SAVE TRAIN IMAGES (with augmentation)
# ======================
print("\nSaving TRAIN images with augmentation...")
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
    img_path = row[image_column]
    label = row[label_column]
    
    try:
        img = Image.open(img_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Save multiple augmented copies
        for i in range(augmentations_per_image):
            aug_img = train_transform(img)
            out_path = os.path.join(
                train_dir, f"cluster_{label}",
                f"{base_name}_aug{i}.jpg"
            )
            aug_img.save(out_path)

    except Exception as e:
        print(f"\nFailed loading {img_path}: {e}")


# ======================
# SAVE VALIDATION IMAGES (no augmentation)
# ======================
print("\nSaving VALIDATION images (no augmentation)...")
for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validation"):
    img_path = row[image_column]
    label = row[label_column]
    
    try:
        img = Image.open(img_path).convert("RGB")
        img = val_test_transform(img)

        out_path = os.path.join(
            val_dir, f"cluster_{label}",
            os.path.basename(img_path)
        )
        img.save(out_path)

    except Exception as e:
        print(f"\nFailed loading {img_path}: {e}")


# ======================
# SAVE TEST IMAGES (no augmentation)
# ======================
print("\nSaving TEST images (no augmentation)...")
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
    img_path = row[image_column]
    label = row[label_column]
    
    try:
        img = Image.open(img_path).convert("RGB")
        img = val_test_transform(img)

        out_path = os.path.join(
            test_dir, f"cluster_{label}",
            os.path.basename(img_path)
        )
        img.save(out_path)

    except Exception as e:
        print(f"\nFailed loading {img_path}: {e}")


# ======================
# SAVE SPLIT METADATA
# ======================
metadata = {
    "train": train_df[[image_column, label_column]],
    "val": val_df[[image_column, label_column]],
    "test": test_df[[image_column, label_column]]
}

for split_name, split_df in metadata.items():
    split_df.to_csv(os.path.join(output_root, f"{split_name}_metadata.csv"), index=False)

print("\n" + "="*50)
print("✅ Dataset successfully prepared for CNN!")
print("="*50)
print(f"Training folder  : {train_dir}")
print(f"Validation folder: {val_dir}")
print(f"Testing folder   : {test_dir}")
print(f"\nMetadata files saved in: {output_root}")
print("="*50)