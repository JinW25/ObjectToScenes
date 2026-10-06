"""Clutter-level classifier: predicts C0 / C1 / C2 for the target object in a scene image.

Input: a top-down RGB scene image in which the target object is marked with a green
bounding box (RGB (50, 255, 50)). From it the classifier uses

  * the image (grayscale, 224x224, CLAHE-equalised, per-image normalised), and
  * 6 spatial features of the target box (``SPATIAL_FEATURES``).

The model is a ResNet (default ResNet50, 1-channel input) image branch and an MLP spatial
branch, fused by fully connected layers. Weights: ``weights/classifier/best_model.pth`` with
its ``config.json``; ``classifier/training.py`` reproduces them.

This module does not depend on Isaac Sim, so it can classify images from any simulator or
from a real camera.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .clutter_levels import apply_complexity_correction

SPATIAL_FEATURES = ["obj_x", "obj_y", "distance_to_center", "bbox_area", "bbox_width", "bbox_height"]
CLASS_NAMES = ["C0_easy", "C1_medium", "C2_hard"]


# ── image preprocessing (identical to training) ───────────────────────────────

class HistogramEqualization:
    """Apply CLAHE to normalize brightness/contrast."""

    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_array = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return Image.fromarray(clahe.apply(img_array))


class AdaptiveNormalization:
    """Normalize each image individually."""

    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        if std < 1e-6:
            std = 1.0
        return (tensor - mean) / std


def image_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        HistogramEqualization(clip_limit=3.0, tile_grid_size=(8, 8)),
        transforms.ToTensor(),
        AdaptiveNormalization(),
    ])


# ── spatial features from the green target box ────────────────────────────────

def bbox_features_from_pixels(x_min, y_min, x_max, y_max, width, height) -> dict:
    """Normalised bounding-box features from pixel coordinates."""
    return {
        "bbox_x_min": x_min / width,
        "bbox_x_max": x_max / width,
        "bbox_y_min": y_min / height,
        "bbox_y_max": y_max / height,
        "bbox_center_x": ((x_min + x_max) / 2) / width,
        "bbox_center_y": ((y_min + y_max) / 2) / height,
        "bbox_width": (x_max - x_min) / width,
        "bbox_height": (y_max - y_min) / height,
        "bbox_area": ((x_max - x_min) * (y_max - y_min)) / (width * height),
    }


def detect_green_bbox_from_image(image_path):
    """Detect the green target bounding box in a scene image; returns normalised bbox features or None."""
    try:
        img = np.array(Image.open(image_path))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        height, width = img_hsv.shape[:2]

        mask = cv2.inRange(img_hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_area = 0
        best_bbox = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 100 and 0.1 < w / h < 10 and area > max_area:
                max_area = area
                best_bbox = (x, y, x + w, y + h)

        if best_bbox is None:
            return None
        return bbox_features_from_pixels(*best_bbox, width, height)

    except Exception as e:
        print(f"[ERROR] Failed to detect bbox from {image_path}: {e}")
        return None


def estimate_3d_position_from_bbox(bbox_features):
    """Approximate normalised object position from its image bounding box."""
    center_x = bbox_features["bbox_center_x"]
    center_y = bbox_features["bbox_center_y"]
    bbox_area = bbox_features["bbox_area"]
    bbox_height = bbox_features["bbox_height"]

    obj_x_norm = center_x
    obj_y_norm = 0.5 + (0.5 - center_y) * 0.8
    obj_z_norm = np.clip((bbox_height * 3.0 + bbox_area * 2.0) / 2.0, 0, 1)

    dist_to_center = np.sqrt((obj_x_norm - 0.5) ** 2 + (obj_y_norm - 0.5) ** 2)
    dist_to_camera = np.clip(center_y * (1.0 - bbox_area * 2.0), 0, 1)

    return {
        "obj_x": np.clip(obj_x_norm, 0, 1),
        "obj_y": np.clip(obj_y_norm, 0, 1),
        "obj_z": obj_z_norm,
        "distance_to_center": np.clip(dist_to_center, 0, 1),
        "distance_to_camera": np.clip(dist_to_camera, 0, 1),
    }


def spatial_features_from_bbox(bbox_features) -> np.ndarray:
    """The 6 classifier input features (order = ``SPATIAL_FEATURES``)."""
    position_features = estimate_3d_position_from_bbox(bbox_features)
    return np.array([
        position_features["obj_x"],
        position_features["obj_y"],
        position_features["distance_to_center"],
        bbox_features["bbox_area"],
        bbox_features["bbox_width"],
        bbox_features["bbox_height"],
    ], dtype=np.float32)


def extract_spatial_features_from_image(image_path):
    """6 spatial features of the green target box in ``image_path`` (None if no box is found)."""
    bbox_features = detect_green_bbox_from_image(image_path)
    if bbox_features is None:
        print(f"[WARN] Could not detect bbox in {image_path}")
        return None
    return spatial_features_from_bbox(bbox_features)


# ── model ─────────────────────────────────────────────────────────────────────

class MultiModalCNN(nn.Module):
    def __init__(self, base_model_name, num_classes, num_location_features, dropout_rate=0.35):
        super().__init__()
        from torchvision import models

        if base_model_name not in ("resnet18", "resnet34", "resnet50"):
            raise ValueError(f"Unknown model: {base_model_name}")
        self.cnn = getattr(models, base_model_name)(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        cnn_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

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

        self.fusion = nn.Sequential(
            nn.Linear(cnn_features + 128, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, location):
        combined = torch.cat([self.cnn(image), self.location_branch(location)], dim=1)
        return self.fusion(combined)


def load_classifier_model(model_dir, device):
    """Load ``best_model.pth`` + ``config.json`` from ``model_dir``."""
    model_path = Path(model_dir) / "best_model.pth"
    config_path = Path(model_dir) / "config.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} (run scripts/download_weights.sh)")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    print(f"[CLASSIFIER] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint["model_state_dict"]["fusion.6.weight"].shape[0]

    model = MultiModalCNN(
        base_model_name=config.get("model_name", "resnet50"),
        num_classes=num_classes,
        num_location_features=len(SPATIAL_FEATURES),
        dropout_rate=config.get("dropout_rate", 0.35),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[CLASSIFIER] ✓ Model loaded ({num_classes} classes, {len(SPATIAL_FEATURES)} location features)")
    return model


def predict_complexity_from_image(model, image_path, device, verbose=True) -> int:
    """Raw classifier prediction (0 = C0, 1 = C1, 2 = C2) for a scene image with a green target box.

    Returns 2 (hard) if the box cannot be found or inference fails, as in the paper's runs.
    """
    spatial_features = extract_spatial_features_from_image(image_path)
    if spatial_features is None:
        print(f"[WARN] Could not extract features from {image_path}, defaulting to complexity 2 (Hard)")
        return 2

    if verbose:
        print(f"\n[FEATURES] Extracted from: {Path(image_path).name}")
        for i, (name, value) in enumerate(zip(SPATIAL_FEATURES, spatial_features), 1):
            print(f"  {i}. {name + ':':<21} {value:.3f}")

    try:
        image = Image.open(image_path).convert("L")
        image_tensor = image_transform()(image).unsqueeze(0).to(device)
        if verbose:
            print(f"[IMAGE] Shape: {image_tensor.shape}  "
                  f"Range: [{image_tensor.min().item():.3f}, {image_tensor.max().item():.3f}]")
    except Exception as e:
        print(f"[ERROR] Failed to load/preprocess image: {e}")
        return 2

    spatial_tensor = torch.FloatTensor(spatial_features).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            outputs = model(image_tensor, spatial_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            complexity = torch.max(outputs, 1)[1].item()
        if verbose:
            print(f"\n[PREDICTION]  Predicted: C{complexity}  Confidence: {probabilities[complexity].item():.2%}")
            for k, name in enumerate(CLASS_NAMES):
                print(f"  {name:<10} {probabilities[k].item():.2%}")
        return complexity
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return 2


def classify_scene(model, image_path, neighbor_count: int, device, obj_name="target", verbose=True) -> int:
    """Protocol clutter level of a scene: classifier prediction corrected by the neighbour count."""
    predicted = predict_complexity_from_image(model, image_path, device, verbose=verbose)
    return apply_complexity_correction(predicted, neighbor_count, obj_name=obj_name)
