# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GG-CNN (original, RSS 2018) architecture + inference helper.

The class below is the user's exact provided GGCNN definition, unmodified
(filter_sizes=[32,16,8,8,16,32], kernel_sizes=[9,5,3,3,5,9], strides=[3,2,2,2,2,3]),
verified with `load_state_dict(..., strict=True)` against
ggcnn_epoch_23_cornell_statedict.pt -- zero missing/unexpected keys. Also verified
empirically that at a 300x300 input this architecture's own conv/deconv math
naturally produces exactly 300x300 output maps (the kernel_size=2 output heads
have no padding, but by the time they run the preceding conv/deconv stack has
already expanded to 301x301, so heads shrinking by 1px lands exactly back on
300x300) -- no cropping/padding fixup needed, unlike some other GG-CNN variants.
If you load a DIFFERENT checkpoint or change input_channels/resolution, re-verify
both of these the same way.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_FILTER_SIZES = [32, 16, 8, 8, 16, 32]
_KERNEL_SIZES = [9, 5, 3, 3, 5, 9]
_STRIDES = [3, 2, 2, 2, 2, 3]


class GGCNN(nn.Module):
    """GG-CNN. Equivalent to the Keras model used in the RSS paper
    (https://arxiv.org/abs/1804.05172)."""

    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, _FILTER_SIZES[0], _KERNEL_SIZES[0], stride=_STRIDES[0], padding=3)
        self.conv2 = nn.Conv2d(_FILTER_SIZES[0], _FILTER_SIZES[1], _KERNEL_SIZES[1], stride=_STRIDES[1], padding=2)
        self.conv3 = nn.Conv2d(_FILTER_SIZES[1], _FILTER_SIZES[2], _KERNEL_SIZES[2], stride=_STRIDES[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(_FILTER_SIZES[2], _FILTER_SIZES[3], _KERNEL_SIZES[3],
                                          stride=_STRIDES[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(_FILTER_SIZES[3], _FILTER_SIZES[4], _KERNEL_SIZES[4],
                                          stride=_STRIDES[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(_FILTER_SIZES[4], _FILTER_SIZES[5], _KERNEL_SIZES[5],
                                          stride=_STRIDES[5], padding=3, output_padding=1)
        self.pos_output = nn.Conv2d(_FILTER_SIZES[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(_FILTER_SIZES[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(_FILTER_SIZES[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(_FILTER_SIZES[5], 1, kernel_size=2)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)
        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {'p_loss': p_loss, 'cos_loss': cos_loss, 'sin_loss': sin_loss, 'width_loss': width_loss},
            'pred': {'pos': pos_pred, 'cos': cos_pred, 'sin': sin_pred, 'width': width_pred},
        }


def _gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 2.0) -> torch.Tensor:
    """Small self-contained Gaussian blur (avoids adding a scipy dependency) --
    matches the original GG-CNN repo's post-processing smoothing of the quality
    map before taking the argmax, which reduces sensitivity to single-pixel noise.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    kernel2d = torch.outer(g1d, g1d).view(1, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(img, kernel2d, padding=pad)


@torch.no_grad()
def predict_grasp_candidates(
    model: GGCNN,
    depth_crop: torch.Tensor,
    device: str = "cuda",
    top_k: int = 10,
    nms_kernel: int = 15,
) -> list[dict]:
    """Run GG-CNN on a preprocessed depth crop and return up to `top_k` candidate
    grasps (local maxima of the quality map, ranked by quality), instead of a
    single global best. No masking is applied here -- the caller is expected to
    pick among these candidates using its own criteria (e.g. which one lands
    closest to a known target coordinate), which is what actually enforces
    "grasp the target, not the clutter" one level up.

    Args:
        depth_crop: (H, W) preprocessed depth image, same convention as
            `predict_best_grasp`.
        top_k: max number of candidates to return.
        nms_kernel: local-maximum window size (pixels) for simple non-max
            suppression -- a pixel counts as a candidate only if its quality
            equals the max within this window around it.

    Returns:
        List of dicts (row, col, angle_rad, width, quality), sorted by quality
        descending, length <= top_k. Empty list if no positive-quality pixels
        exist at all.
    """
    x = depth_crop.to(device).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pos_out, cos_out, sin_out, width_out = model(x)

    quality = torch.sigmoid(pos_out)
    quality = _gaussian_blur(quality)
    angle = torch.atan2(sin_out, cos_out) / 2.0
    width = torch.sigmoid(width_out)

    q = quality[0, 0]
    pooled = F.max_pool2d(
        q.unsqueeze(0).unsqueeze(0), kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2
    )[0, 0]
    is_peak = (q == pooled) & (q > 0.0)

    rows, cols = torch.where(is_peak)
    candidates = [
        {
            "row": int(r),
            "col": int(c),
            "angle_rad": float(angle[0, 0, r, c]),
            "width": float(width[0, 0, r, c]),
            "quality": float(q[r, c]),
        }
        for r, c in zip(rows.tolist(), cols.tolist())
    ]
    candidates.sort(key=lambda d: -d["quality"])
    return candidates[:top_k]


@torch.no_grad()
def predict_best_grasp(
    model: GGCNN,
    depth_crop: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    device: str = "cuda",
) -> dict:
    """Run GG-CNN on a preprocessed depth crop and return the single best grasp.

    Args:
        depth_crop: (H, W) depth image, ALREADY cropped/resized to the network's
            input resolution and normalized (mean-subtracted, roughly in [-1, 1] --
            standard GG-CNN preprocessing). H and W must match what conv/deconv
            stride math expects (the original network was trained at 300x300;
            other sizes divisible by 12 should still run but were not what it was
            trained on).
        valid_mask: optional (H, W) boolean mask, same resolution as depth_crop --
            True where the TARGET object is. When given, the argmax is restricted
            to these pixels so a grasp can only be selected on the target, not on
            clutter that happens to also be in frame.
        device: torch device string.

    Returns:
        dict with pixel (row, col) as ints, angle_rad, width (normalized network
        output, 0-1 range, caller converts to meters), and quality (0-1).
    """
    x = depth_crop.to(device).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pos_out, cos_out, sin_out, width_out = model(x)

    quality = torch.sigmoid(pos_out)
    quality = _gaussian_blur(quality)
    angle = torch.atan2(sin_out, cos_out) / 2.0
    width = torch.sigmoid(width_out)

    quality_2d = quality[0, 0]
    if valid_mask is not None:
        # Anything outside the target mask is disqualified, not just down-weighted
        # -- prevents the argmax from ever landing on a clutter object even if it
        # happens to score a higher raw quality than the (possibly partially
        # occluded) target.
        quality_2d = torch.where(valid_mask.to(device), quality_2d, torch.zeros_like(quality_2d))

    if float(quality_2d.max()) <= 0.0:
        return {"row": None, "col": None, "angle_rad": 0.0, "width": 0.5, "quality": 0.0}

    flat_idx = int(torch.argmax(quality_2d))
    row, col = divmod(flat_idx, quality_2d.shape[-1])

    return {
        "row": row,
        "col": col,
        "angle_rad": float(angle[0, 0, row, col]),
        "width": float(width[0, 0, row, col]),
        "quality": float(quality_2d[row, col]),
    }