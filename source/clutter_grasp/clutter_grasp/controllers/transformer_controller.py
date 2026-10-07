"""Distilled transformer controller for the Contactile benchmark environment.

One transformer for all objects (distilled from the per-object PPO policies). It reads the
environment's 234-dim observation, normalises it with the statistics saved during distillation,
replaces the observation's point-cloud slice with the target's live mesh points, and outputs the
12-dim hand action in [-1, 1].

Checkpoints (weights/transformer/<name>/):
    best_model.pth        {"model": state_dict, "val_loss": ...}
    obs_norm_stats.npz    {"mean": (234,), "std": (234,)}
    config.json           architecture (d_model, nhead, num_layers, dim_feedforward, num_proprio_tokens)
"""

import json
from pathlib import Path

import numpy as np
import torch

from .transformer_model import UniGraspTransformer, get_proprio_obs_dim, split_obs

ARCH_DEFAULTS = {"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512, "num_proprio_tokens": 4}


class TransformerController:
    def __init__(self, env, checkpoint, obs_norm_stats=None, **arch):
        """env: the unwrapped BenchmarkEnv. arch: architecture overrides (None values are ignored);
        otherwise taken from config.json next to the checkpoint, else ARCH_DEFAULTS."""
        self.env = env
        checkpoint = Path(checkpoint)
        obs_norm_stats = Path(obs_norm_stats) if obs_norm_stats else checkpoint.parent / "obs_norm_stats.npz"
        for f in (checkpoint, obs_norm_stats):
            if not f.exists():
                raise FileNotFoundError(f"{f} not found (run scripts/download_weights.sh transformer)")

        cfg = dict(ARCH_DEFAULTS)
        cfg_file = checkpoint.parent / "config.json"
        if cfg_file.exists():
            saved = json.loads(cfg_file.read_text())
            cfg.update({k: saved[k] for k in ARCH_DEFAULTS if k in saved})
        cfg.update({k: v for k, v in arch.items() if v is not None})

        stats = np.load(str(obs_norm_stats))
        self.obs_mean = stats["mean"].astype(np.float32)
        self.obs_std = stats["std"].astype(np.float32)

        device = env.device
        obs_dim = env.cfg.observation_space        # 234
        self.num_pc_points = env.cfg.num_object_pc_points  # 32
        action_dim = env.cfg.action_space          # 12
        self.model = UniGraspTransformer(
            obs_dim=get_proprio_obs_dim(obs_dim, self.num_pc_points),
            action_dim=action_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_encoder_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            num_proprio_tokens=cfg["num_proprio_tokens"],
        ).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[TRANSFORMER] ✓ Loaded {checkpoint}  val_loss={ckpt.get('val_loss', float('nan')):.6f}  arch={cfg}")

    def act(self, obs):
        # 1. normalise with the distillation statistics
        obs_norm = np.clip((np.asarray(obs, dtype=np.float32) - self.obs_mean) / self.obs_std, -10.0, 10.0)
        obs_t = torch.tensor(obs_norm.astype(np.float32), device=self.env.device)
        # 2. proprioceptive tokens; the observation's point-cloud slice is replaced by the live mesh
        proprio_t, _ = split_obs(obs_t, self.num_pc_points)
        pc_t = self.env._object_mesh_points_local.unsqueeze(0).expand(self.env.num_envs, -1, -1)
        # 3. forward pass -> clipped actions
        with torch.inference_mode():
            return np.clip(self.model(proprio_t, pc_t).cpu().numpy(), -1.0, 1.0)
