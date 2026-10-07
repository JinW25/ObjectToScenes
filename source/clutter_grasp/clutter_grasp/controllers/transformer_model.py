# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
UniGraspTransformer Model + Dataset
====================================
Implements the student transformer that is distilled from per-object RL teachers.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class GraspTrajectoryDataset(Dataset):
    """
    Loads one .npz trajectory file and returns (obs, pc, action) tuples.

    Each sample is a single time-step drawn from a successful trajectory.
    The point cloud is constant across all steps of the same trajectory.

    .npz schema:
        observations : (N, T, obs_dim)  float32  — RAW unnormalized obs
        actions      : (N, T, act_dim)  float32  — Tanh output in [-1, 1]
        point_clouds : (N, P, 3)        float32  — LOCAL object frame
        lengths      : (N,)             int32
        mask         : (N, T)           bool

    obs_mean / obs_std: global normalization stats computed by 3_distill_train.py.
    If provided, obs are normalized to ~N(0,1) and clipped to ±10 here.
    """

    def __init__(
        self,
        npz_path: str | Path,
        obs_indices: slice | None = None,
        augment_pc: bool = True,
        pc_noise_std: float = 0.001,
        obs_mean: np.ndarray | None = None,
        obs_std:  np.ndarray | None = None,
    ):
        super().__init__()
        data = np.load(npz_path)
        self.observations = data["observations"]  # (N, T, D)
        self.actions      = data["actions"]       # (N, T, A)
        self.point_clouds = data["point_clouds"]  # (N, P, 3)
        self.mask         = data["mask"]          # (N, T)
        self.lengths      = data["lengths"]       # (N,)
        self.object_name  = str(data["object_name"][0]) if "object_name" in data else "unknown"

        # Normalize raw obs → ~N(0,1), clip to ±10 (matches VecNormalize behavior).
        # obs_mean/std are computed globally across ALL objects in 3_distill_train.py
        # and saved to obs_norm_stats.npz — eval loads the same file.
        if obs_mean is not None and obs_std is not None:
            self.observations = np.clip(
                (self.observations - obs_mean) / obs_std, -10.0, 10.0
            ).astype(np.float32)

        # Build flat index: (traj_idx, step_idx) for every valid step
        self._index: list[tuple[int, int]] = []
        for traj_i, length in enumerate(self.lengths):
            for step_j in range(int(length)):
                self._index.append((traj_i, step_j))

        self.obs_indices  = obs_indices
        self.augment_pc   = augment_pc
        self.pc_noise_std = pc_noise_std

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        traj_i, step_j = self._index[idx]

        obs = torch.from_numpy(self.observations[traj_i, step_j].copy())  # (D,)
        act = torch.from_numpy(self.actions[traj_i, step_j].copy())       # (A,)
        pc  = torch.from_numpy(self.point_clouds[traj_i].copy())          # (P,3)

        if self.obs_indices is not None:
            obs = obs[self.obs_indices]

        if self.augment_pc and self.pc_noise_std > 0:
            pc = pc + torch.randn_like(pc) * self.pc_noise_std

        return obs, pc, act


def build_dataset(
    trajectory_dir: str | Path,
    obs_mean: np.ndarray | None = None,
    obs_std:  np.ndarray | None = None,
    **dataset_kwargs,
) -> ConcatDataset:
    """Load all .npz files in trajectory_dir and concatenate.

    obs_mean / obs_std: if provided, passed to each GraspTrajectoryDataset
    so obs are normalized consistently across all objects.
    """
    traj_dir  = Path(trajectory_dir)
    npz_files = sorted(traj_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {traj_dir}")

    datasets = []
    for f in npz_files:
        ds = GraspTrajectoryDataset(
            f,
            obs_mean=obs_mean,
            obs_std=obs_std,
            **dataset_kwargs,
        )
        print(f"  Loaded {f.stem}: {len(ds):,} steps")
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    print(f"  Total: {len(combined):,} steps from {len(datasets)} objects")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class PointCloudEncoder(nn.Module):
    """Mini-PointNet: shared MLP → global max-pool → token. (B,P,3) → (B,d_model)"""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        feat  = self.mlp(pc)          # (B, P, d_model)
        token = feat.max(dim=1)[0]    # global max pool → (B, d_model)
        return token


class ProprioEncoder(nn.Module):
    """Splits proprioception into num_tokens chunks, encodes each → (B, K, d_model)."""

    def __init__(self, obs_dim: int, d_model: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        chunk = obs_dim // num_tokens
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(chunk if i < num_tokens - 1 else obs_dim - chunk * (num_tokens - 1), d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for i in range(num_tokens)
        ])
        self._chunk = chunk

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = []
        for i, enc in enumerate(self.encoders):
            start = i * self._chunk
            end   = start + self._chunk if i < self.num_tokens - 1 else obs.shape[-1]
            tokens.append(enc(obs[:, start:end]))
        return torch.stack(tokens, dim=1)  # (B, K, d_model)


class UniGraspTransformer(nn.Module):
    """
    UniGraspTransformer student network.

    Token sequence: [object_token | proprio_0 | ... | proprio_K]
    Action head reads from object token (index 0) after transformer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 12,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_proprio_tokens: int = 4,
        pc_hidden: int = 128,
    ):
        super().__init__()
        self.d_model    = d_model
        self.action_dim = action_dim

        self.pc_encoder      = PointCloudEncoder(d_model, hidden=pc_hidden)
        self.proprio_encoder = ProprioEncoder(obs_dim, d_model, num_tokens=num_proprio_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs : (B, obs_dim)  normalized proprioception
            pc  : (B, P, 3)     object point cloud in LOCAL frame
        Returns:
            action : (B, action_dim) in [-1, 1]
        """
        obj_token      = self.pc_encoder(pc).unsqueeze(1)          # (B, 1, d_model)
        proprio_tokens = self.proprio_encoder(obs)                  # (B, K, d_model)
        tokens         = torch.cat([obj_token, proprio_tokens], dim=1)  # (B, 1+K, d_model)
        out            = self.transformer(tokens)                   # (B, 1+K, d_model)
        action         = self.action_head(out[:, 0, :])             # (B, action_dim)
        return action

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, pc)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  OBSERVATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_proprio_obs_dim(full_obs_dim: int, num_pc_points: int = 32) -> int:
    """Proprioception dim = full obs dim minus the embedded point cloud chunk."""
    return full_obs_dim - num_pc_points * 3


def split_obs(obs: torch.Tensor, num_pc_points: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split full obs into (proprio, pc_flat).

    Full obs layout:
        hand_pos         3
        hand_quat        4
        hand_lin_vel     3
        hand_ang_vel     3
        joint_pos        6
        joint_vel        6
        finger_site_pos 30
        palm_site_pos   21
        object_pc_flat  num_pc_points*3   ← stripped out here
        chamfer_dist    17
        contact_forces  17
        object_pos       3
        object_quat      4
        object_lin_vel   3
        object_ang_vel   3
        object_init_pos  3
        actions         12
    """
    pc_start = 3 + 4 + 3 + 3 + 6 + 6 + 30 + 21  # = 76
    pc_end   = pc_start + num_pc_points * 3

    pc_flat = obs[..., pc_start:pc_end]
    proprio = torch.cat([obs[..., :pc_start], obs[..., pc_end:]], dim=-1)
    return proprio, pc_flat