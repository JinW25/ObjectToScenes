import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import cv2
import argparse
from pathlib import Path

# ======================
# CUSTOM TRANSFORMS FOR COLOR-INVARIANCE
# ======================
class HistogramEqualization:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    This normalizes brightness/contrast across all images,
    making dark tables and bright tables look similar.
    """
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        """Apply CLAHE to PIL Image"""
        # Convert PIL to numpy array
        img_array = np.array(img)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=self.tile_grid_size
        )
        
        # Apply equalization
        equalized = clahe.apply(img_array)
        
        # Convert back to PIL Image
        return Image.fromarray(equalized)


class AdaptiveNormalization:
    """Normalize each image individually to zero mean and unit variance
    
    This ensures each image has consistent statistics regardless of 
    original brightness/contrast.
    """
    def __call__(self, tensor):
        """tensor shape: (C, H, W)"""
        mean = tensor.mean()
        std = tensor.std()
        
        # Avoid division by zero
        if std < 1e-6:
            std = 1.0
        
        # Normalize to mean=0, std=1
        normalized = (tensor - mean) / std
        return normalized


class RandomGammaCorrection:
    """Apply random gamma correction to simulate different exposures"""
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range
    
    def __call__(self, img):
        """Apply random gamma to PIL Image"""
        gamma = np.random.uniform(*self.gamma_range)
        return transforms.functional.adjust_gamma(img, gamma=gamma)


# ======================
# CONFIG
# ======================
# Repository root (classifier/ -> repo) and git-ignored data directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Spatial-feature order assumed by the benchmark's classifier loader; must match
# clutter_grasp.protocol.classifier.SPATIAL_FEATURES (not imported so this script runs
# without the Isaac Lab package installed).
BENCHMARK_FEATURES = ['obj_x', 'obj_y', 'distance_to_center', 'bbox_area', 'bbox_width', 'bbox_height']

parser = argparse.ArgumentParser(
    description="Train the multimodal (image + spatial features) clutter-complexity classifier (step 5). "
                "Writes best_model.pth + config.json in the format loaded by the benchmark.")
parser.add_argument("--data_dir", type=str, default=str(DATA_DIR / "classifier" / "cnn_dataset"),
                    help="Split dataset from preprocessing.py (train/ val/ test/ + *_metadata.csv)")
parser.add_argument("--csv_path", type=str,
                    default=str(DATA_DIR / "classifier" / "pca_kmeans" / "pca_kmeans_results.csv"),
                    help="CSV with an 'image_pixels' column and the spatial features "
                         "(pca_kmeans_results.csv works for both the MuJoCo and the Isaac data path)")
parser.add_argument("--output_dir", type=str, default=str(DATA_DIR / "classifier" / "training"),
                    help="Where best_model.pth, config.json and plots are written")
parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
parser.add_argument("--num_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

config = {
    "data_dir": args.data_dir,
    "csv_path": args.csv_path,
    "output_dir": args.output_dir,
    "img_size": 224,
    "batch_size": args.batch_size,
    "num_epochs": args.num_epochs,  # Train longer with strong augmentation
    "learning_rate": 0.0001,  # Slightly lower for stability
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": args.num_workers,
    
    # Selected features (6 total) - NOW INCLUDING distance_to_center
    "selected_features": [
        'obj_x',
        'obj_y',
        'distance_to_center',  # ADDED
        'bbox_area',
        'bbox_width',
        'bbox_height',
    ],
    
    "model_name": args.model_name,
    "early_stopping_patience": 20,  # More patience with augmentation
    "weight_decay": 5e-4,  # Stronger regularization
    "dropout_rate": 0.45,  # Increased dropout
    "label_smoothing": 0.12,  # More smoothing
}

os.makedirs(config["output_dir"], exist_ok=True)

if config["selected_features"] != BENCHMARK_FEATURES:
    print(f"[WARN] selected_features differ from the order the benchmark loader assumes "
          f"({BENCHMARK_FEATURES}); the resulting model will not be loadable by the benchmark as-is.")

# config.json is read by the benchmark loader (model_name, dropout_rate). Same keys as the
# released weights' config.json, plus selected_features for reference.
with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
    json.dump({k: v for k, v in config.items() if not isinstance(v, pd.DataFrame)}, f, indent=4)

print("="*70)
print("COLOR-INVARIANT MULTI-MODAL TRAINING")
print("="*70)
print(f"Selected features ({len(config['selected_features'])}):")
for i, feat in enumerate(config['selected_features'], 1):
    print(f"  {i}. {feat}")
print("="*70)
print("\nColor-Invariance Features:")
print("  ✓ Histogram Equalization (CLAHE)")
print("  ✓ Adaptive Normalization")
print("  ✓ Strong Brightness/Contrast Augmentation")
print("  ✓ Random Gamma Correction")
print("  ✓ Random Color Inversion")
print("="*70)


# ======================
# DATASET WITH SELECTED FEATURES
# ======================
class MultiModalDataset(Dataset):
    def __init__(self, root_dir, csv_path, split_csv_path, transform=None, selected_features=None):
        self.root_dir = root_dir
        self.transform = transform
        self.selected_features = selected_features or []
        self.samples = []
        
        # Load split metadata
        split_df = pd.read_csv(split_csv_path)
        
        # Load main CSV with all features
        main_df = pd.read_csv(csv_path)
        
        # Create mapping from image_path to features
        self.img_to_features = {}
        self.name_to_features = {}  # fallback keyed by file name (paths may differ in form)
        for _, row in main_df.iterrows():
            img_path = row['image_pixels']
            self.img_to_features[img_path] = self.name_to_features[os.path.basename(str(img_path))] = {
                'bbox_center_x': row.get('bbox_center_x', 0.5),
                'bbox_center_y': row.get('bbox_center_y', 0.5),
                'bbox_width': row.get('bbox_width', 0.15),
                'bbox_height': row.get('bbox_height', 0.15),
                'bbox_area': row.get('bbox_area', 0.0225),
                'obj_x': row.get('obj_x', 0.5),
                'obj_y': row.get('obj_y', 0.5),
                'obj_z': row.get('obj_z', 0.5),
                'distance_to_center': row.get('distance_to_center', 0.3),
                'distance_to_camera': row.get('distance_to_camera', 0.5),
            }
        
        # Get class info from directory structure
        self.class_names = sorted([d for d in os.listdir(root_dir) 
                                   if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load samples from split
        for _, row in split_df.iterrows():
            original_img_path = row['image_path']
            label = row['cluster']
            
            # Find augmented versions in the directory
            class_dir = os.path.join(root_dir, f"cluster_{label}")
            if not os.path.isdir(class_dir):
                continue
            
            base_name = os.path.splitext(os.path.basename(original_img_path))[0]
            
            # Find all augmented versions or original
            for img_name in os.listdir(class_dir):
                if img_name.startswith(base_name):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, original_img_path, label))
    
    def get_location_features(self, original_img_path):
        """Extract only the selected features from CSV mapping"""
        features_dict = self.img_to_features.get(original_img_path)
        if features_dict is None:
            features_dict = self.name_to_features.get(os.path.basename(str(original_img_path)), {})
        
        # Extract only selected features in order
        features = []
        for feat_name in self.selected_features:
            # Default values if feature is missing
            default_values = {
                'bbox_area': 0.0225,
                'bbox_width': 0.15,
                'bbox_height': 0.15,
                'obj_x': 0.5,
                'obj_y': 0.5,
                'distance_to_center': 0.3,  # ADDED
            }
            features.append(features_dict.get(feat_name, default_values.get(feat_name, 0.5)))
        
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, original_img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        
        # Get selected location features from original image
        location_features = self.get_location_features(original_img_path)
        
        return image, torch.FloatTensor(location_features), label


# ======================
# MULTI-MODAL MODEL
# ======================
class MultiModalCNN(nn.Module):
    """
    Combines CNN features from images with selected spatial features.
    """
    def __init__(self, base_model_name, num_classes, num_location_features, dropout_rate=0.45):
        super(MultiModalCNN, self).__init__()
        
        # CNN branch for image features
        if base_model_name == "resnet18":
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            cnn_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
            
        elif base_model_name == "resnet34":
            self.cnn = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            cnn_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
            
        elif base_model_name == "resnet50":
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            cnn_features = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        
        # Location branch (selected spatial features)
        self.location_branch = nn.Sequential(
            nn.Linear(num_location_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
        )
        
        # Fusion layer
        fusion_features = cnn_features + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, location):
        # Extract image features
        img_features = self.cnn(image)
        
        # Extract location features
        loc_features = self.location_branch(location)
        
        # Concatenate and classify
        combined = torch.cat([img_features, loc_features], dim=1)
        output = self.fusion(combined)
        return output


# ======================
# ENHANCED TRANSFORMS (COLOR-INVARIANT)
# ======================
print("\n[INFO] Setting up color-invariant transforms...")

train_transform = transforms.Compose([
    transforms.Resize((int(config["img_size"] * 1.15), int(config["img_size"] * 1.15))),
    
    # ============ STEP 1: Normalize brightness FIRST ============
    HistogramEqualization(clip_limit=3.0, tile_grid_size=(8, 8)),
    
    # ============ STEP 2: Geometric augmentations ============
    transforms.RandomResizedCrop(config["img_size"], scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    
    # ============ STEP 3: AGGRESSIVE brightness/contrast variation ============
    transforms.ColorJitter(brightness=0.8, contrast=0.8),
    
    # ============ STEP 4: Random gamma correction (simulates different exposures) ============
    transforms.RandomApply([RandomGammaCorrection(gamma_range=(0.5, 2.0))], p=0.5),
    
    # ============ STEP 5: Random inversion (dark becomes bright, bright becomes dark) ============
    transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.invert(x))], p=0.2),
    
    # ============ STEP 6: Convert to tensor ============
    transforms.ToTensor(),
    
    # ============ STEP 7: Adaptive normalization (per-image) ============
    AdaptiveNormalization(),
    
    # ============ STEP 8: Random erasing ============
    transforms.RandomErasing(p=0.25),
])

val_test_transform = transforms.Compose([
    transforms.Resize((config["img_size"], config["img_size"])),
    
    # ============ CRITICAL: Apply same histogram equalization to test data ============
    HistogramEqualization(clip_limit=3.0, tile_grid_size=(8, 8)),
    
    transforms.ToTensor(),
    
    # ============ Use same adaptive normalization ============
    AdaptiveNormalization(),
])

print("[INFO] ✓ Color-invariant transforms configured")


# ======================
# LOAD DATASETS
# ======================
print("\n[INFO] Loading datasets...")

train_dataset = MultiModalDataset(
    os.path.join(config["data_dir"], "train"),
    config["csv_path"],
    os.path.join(config["data_dir"], "train_metadata.csv"),
    transform=train_transform,
    selected_features=config["selected_features"]
)

val_dataset = MultiModalDataset(
    os.path.join(config["data_dir"], "val"),
    config["csv_path"],
    os.path.join(config["data_dir"], "val_metadata.csv"),
    transform=val_test_transform,
    selected_features=config["selected_features"]
)

test_dataset = MultiModalDataset(
    os.path.join(config["data_dir"], "test"),
    config["csv_path"],
    os.path.join(config["data_dir"], "test_metadata.csv"),
    transform=val_test_transform,
    selected_features=config["selected_features"]
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                          shuffle=True, num_workers=config["num_workers"], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], 
                        shuffle=False, num_workers=config["num_workers"], pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], 
                         shuffle=False, num_workers=config["num_workers"], pin_memory=True)

num_classes = len(train_dataset.class_names)
num_location_features = len(config["selected_features"])

print(f"\nDataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
print(f"Classes: {train_dataset.class_names}")
print(f"Location features: {num_location_features}\n")


# ======================
# CREATE MODEL
# ======================
model = MultiModalCNN(
    config["model_name"], 
    num_classes,
    num_location_features=num_location_features,
    dropout_rate=config["dropout_rate"]
)
model = model.to(config["device"])

print(f"Model: {config['model_name']} with {num_location_features} location features")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["learning_rate"] * 8,
    epochs=config["num_epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=0.3
)


# ======================
# TRAINING FUNCTIONS
# ======================
def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, locations, labels in pbar:
        images = images.to(device)
        locations = locations.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, locations)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}'})
    
    return running_loss / total, 100 * correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, locations, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            locations = locations.to(device)
            labels = labels.to(device)
            
            outputs = model(images, locations)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / total, 100 * correct / total


# ======================
# TRAINING LOOP
# ======================
print("Starting color-invariant training...\n")

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
patience_counter = 0

for epoch in range(config["num_epochs"]):
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print("-" * 60)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, config["device"])
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, config["device"])
    
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    
    print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'config': config,
        }, os.path.join(config["output_dir"], "best_model.pth"))
        print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{config['early_stopping_patience']}")
    
    if patience_counter >= config["early_stopping_patience"]:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break
    print()

np.save(os.path.join(config["output_dir"], "history.npy"), history)


# ======================
# PLOT TRAINING CURVES
# ======================
print("\nGenerating training curves...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Training & Validation Loss
ax1 = fig.add_subplot(gs[0, 0])
epochs = range(1, len(history["train_loss"]) + 1)
ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss', linewidth=2, markersize=4)
ax1.plot(epochs, history["val_loss"], 'r-s', label='Val Loss', linewidth=2, markersize=4)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

best_epoch = np.argmax(history["val_acc"])
ax1.axvline(x=best_epoch + 1, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch + 1})')
ax1.legend(fontsize=11)

# 2. Training & Validation Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, history["train_acc"], 'b-o', label='Train Acc', linewidth=2, markersize=4)
ax2.plot(epochs, history["val_acc"], 'r-s', label='Val Acc', linewidth=2, markersize=4)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

ax2.axvline(x=best_epoch + 1, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best Epoch ({best_epoch + 1})')
ax2.legend(fontsize=11)

# 3. Overfitting Gap (Loss)
ax3 = fig.add_subplot(gs[1, 0])
loss_gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
ax3.plot(epochs, loss_gap, 'purple', linewidth=2, marker='o', markersize=4)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.fill_between(epochs, 0, loss_gap, where=(loss_gap > 0), color='red', alpha=0.3, label='Overfitting')
ax3.fill_between(epochs, 0, loss_gap, where=(loss_gap <= 0), color='green', alpha=0.3, label='Underfitting')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Val Loss - Train Loss', fontsize=12)
ax3.set_title('Overfitting Gap (Loss)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Overfitting Gap (Accuracy)
ax4 = fig.add_subplot(gs[1, 1])
acc_gap = np.array(history["train_acc"]) - np.array(history["val_acc"])
ax4.plot(epochs, acc_gap, 'orange', linewidth=2, marker='o', markersize=4)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.fill_between(epochs, 0, acc_gap, where=(acc_gap > 0), color='red', alpha=0.3, label='Overfitting')
ax4.fill_between(epochs, 0, acc_gap, where=(acc_gap <= 0), color='green', alpha=0.3, label='Underfitting')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Train Acc - Val Acc (%)', fontsize=12)
ax4.set_title('Overfitting Gap (Accuracy)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# 5. Loss Comparison
ax5 = fig.add_subplot(gs[2, 0])
final_losses = [history["train_loss"][-1], history["val_loss"][-1]]
colors = ['#3498db', '#e74c3c']
bars = ax5.bar(['Train Loss', 'Val Loss'], final_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('Loss', fontsize=12)
ax5.set_title('Final Epoch Loss Comparison', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, final_losses):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 6. Accuracy Comparison
ax6 = fig.add_subplot(gs[2, 1])
final_accs = [history["train_acc"][-1], history["val_acc"][-1]]
bars = ax6.bar(['Train Acc', 'Val Acc'], final_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('Accuracy (%)', fontsize=12)
ax6.set_title('Final Epoch Accuracy Comparison', fontsize=14, fontweight='bold')
ax6.set_ylim([0, 100])
ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, final_accs):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

fig.suptitle(f'Color-Invariant Training (6 Features with distance_to_center) - Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch + 1})', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(os.path.join(config["output_dir"], "training_curves.png"), dpi=150, bbox_inches='tight')
print(f"✓ Saved training curves to {os.path.join(config['output_dir'], 'training_curves.png')}")
plt.close()


# ======================
# SAVE TRAINING SUMMARY
# ======================
summary_path = os.path.join(config["output_dir"], "training_summary.txt")
with open(summary_path, "w") as f:
    f.write("="*70 + "\n")
    f.write("TRAINING SUMMARY - 6 SELECTED FEATURES (INCLUDING distance_to_center)\n")
    f.write("="*70 + "\n\n")
    
    f.write("Selected Features:\n")
    for i, feat in enumerate(config['selected_features'], 1):
        f.write(f"  {i}. {feat}\n")
    f.write("\n")
    
    f.write("Configuration:\n")
    f.write(f"  Model: {config['model_name']}\n")
    f.write(f"  Batch Size: {config['batch_size']}\n")
    f.write(f"  Learning Rate: {config['learning_rate']}\n")
    f.write(f"  Weight Decay: {config['weight_decay']}\n")
    f.write(f"  Dropout: {config['dropout_rate']}\n")
    f.write(f"  Label Smoothing: {config['label_smoothing']}\n\n")
    
    f.write("Training Progress:\n")
    f.write(f"  Total Epochs: {len(history['train_loss'])}\n")
    f.write(f"  Best Epoch: {best_epoch + 1}\n")
    f.write(f"  Best Val Accuracy: {best_val_acc:.2f}%\n\n")
    
    f.write("Initial Performance (Epoch 1):\n")
    f.write(f"  Train Loss: {history['train_loss'][0]:.4f}, Train Acc: {history['train_acc'][0]:.2f}%\n")
    f.write(f"  Val Loss: {history['val_loss'][0]:.4f}, Val Acc: {history['val_acc'][0]:.2f}%\n\n")
    
    f.write("Final Performance (Last Epoch):\n")
    f.write(f"  Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.2f}%\n")
    f.write(f"  Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_acc'][-1]:.2f}%\n\n")
    
    f.write("Best Performance:\n")
    f.write(f"  Val Loss: {history['val_loss'][best_epoch]:.4f}, Val Acc: {history['val_acc'][best_epoch]:.2f}%\n\n")
    
    f.write("Improvement:\n")
    f.write(f"  Train Acc: {history['train_acc'][0]:.2f}% → {history['train_acc'][-1]:.2f}% (+{history['train_acc'][-1] - history['train_acc'][0]:.2f}%)\n")
    f.write(f"  Val Acc: {history['val_acc'][0]:.2f}% → {best_val_acc:.2f}% (+{best_val_acc - history['val_acc'][0]:.2f}%)\n\n")
    
    f.write("Overfitting Analysis:\n")
    final_gap = history['train_acc'][-1] - history['val_acc'][-1]
    f.write(f"  Final Acc Gap (Train - Val): {final_gap:.2f}%\n")
    if final_gap > 10:
        f.write(f"  Status: High overfitting detected\n")
    elif final_gap > 5:
        f.write(f"  Status: Moderate overfitting\n")
    else:
        f.write(f"  Status: Good generalization\n")
    f.write("\n")
    
    f.write("="*70 + "\n")

print(f"✓ Saved training summary to {summary_path}\n")


# ======================
# EVALUATION
# ======================
print("\n" + "="*70)
print("Evaluating on test set...")
print("="*70)

checkpoint = torch.load(os.path.join(config["output_dir"], "best_model.pth"), map_location=config["device"])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, locations, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(config["device"])
        locations = locations.to(config["device"])
        outputs = model(images, locations)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Saved for later analysis (confusion matrix / classification report)
np.save(os.path.join(config["output_dir"], "test_predictions.npy"), np.array(all_preds))
np.save(os.path.join(config["output_dir"], "test_labels.npy"), np.array(all_labels))

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Best Val Accuracy: {best_val_acc:.2f}%\n")

report = classification_report(all_labels, all_preds, 
                               target_names=train_dataset.class_names, digits=4)
print(report)

with open(os.path.join(config["output_dir"], "results.txt"), "w") as f:
    f.write(f"Multi-Modal Model (Image + 6 Selected Spatial Features with distance_to_center)\n\n")
    f.write("Selected Features:\n")
    for i, feat in enumerate(config['selected_features'], 1):
        f.write(f"  {i}. {feat}\n")
    f.write(f"\nTest Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Best Val Accuracy: {best_val_acc:.2f}%\n\n")
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=train_dataset.class_names,
            yticklabels=train_dataset.class_names, ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=train_dataset.class_names,
            yticklabels=train_dataset.class_names, ax=axes[1])
axes[1].set_title('Normalized Confusion Matrix')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(config["output_dir"], "confusion_matrix.png"), dpi=150)
plt.close()

print(f"\n{'='*70}")
print("✓ Training completed!")
print(f"Results saved in: {config['output_dir']}")
print(f"{'='*70}")