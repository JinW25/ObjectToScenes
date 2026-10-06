#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
collect_classifier_dataset.py
==============================
Collect classifier-training CSV from Isaac Lab DataCollectionEnv,
matching the MuJoCo randomize_scene.py column schema exactly.

Key design principles
---------------------
1. Projection is IDENTICAL to collect_detection_pc_dataset.py:
     rot_mat    = quat_to_rotation_matrix_ros(cam_quat_w_ros)
     points_cam = torch.matmul(points_rel, rot_mat.T.T)   # = @ rot_mat
   cam_pos_w and cam_quat_w are always read from env.camera.data AFTER
   stepping — never computed by hand.

2. Domain randomisation moves the camera PRIM before stepping, so that
   env.camera.data reflects the jittered pose and everything downstream
   (projection, CSV values, validation image) is automatically consistent.

3. All features are computed from the same physics state as the image —
   capture happens first, then AABB/metrics are read.

Output CSV columns (exact sequence)
------------------------------------
  image_pixels, target_grasp_difficulty, target_shape_complexity,
  num_obstacles, num_neighbors, mean_neighbor_distance, min_neighbor_distance,
  mean_neighbor_grasp_difficulty, mean_neighbor_shape_complexity,
  free_space_volume,
  bbox_center_x, bbox_center_y, bbox_width, bbox_height, bbox_area,
  obj_x, obj_y, obj_z, distance_to_center, distance_to_camera,
  bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max

Validation images  (--validate)
---------------------------------
  val/scene_NNNNN_clean.png
  val/scene_NNNNN_metrics.png  — shows:
    lime  box        : target AABB projected bbox
    yellow boxes     : each neighbour AABB, labelled id + distance
    green  dots      : target mesh points projected
    grey   dots      : neighbour mesh points projected
    top-left panel   : all metric values
    bottom-left key  : colour legend

Objects
-------
EGAD objects converted to USD (default ``data/egad_usd``, override with --usd_dir);
see classifier/README.md for the download and conversion to USD.

Usage (from the repository root, or any directory)
-----
python classifier/isaac/collect_classifier_dataset.py \\
    --num_scenes 10 --validate --enable_cameras
python classifier/isaac/collect_classifier_dataset.py \\
    --num_scenes 2000 --enable_cameras --headless

Outputs (default data/classifier/isaac/): classifier_data.csv, images/, validation/.
"""

import argparse
import sys
from pathlib import Path

# This file lives in classifier/isaac/. Paths are computed here because the
# clutter_grasp package cannot be imported before Isaac Sim is launched.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_scenes",      type=int,   default=500)
parser.add_argument("--output_dir",      type=str,   default=str(DATA_DIR / "classifier" / "isaac"),
                    help="Output folder for classifier_data.csv, images/ and validation/ (CSV rows are appended)")
parser.add_argument("--usd_dir",         type=str,   default=str(DATA_DIR / "egad_usd"),
                    help="Folder with the EGAD objects as <id>.usd (see classifier/README.md)")
parser.add_argument("--min_objects",     type=int,   default=5)
parser.add_argument("--max_objects",     type=int,   default=60)
parser.add_argument("--settling_steps",  type=int,   default=150)
parser.add_argument("--neighbor_margin", type=float, default=0.02)
parser.add_argument("--validate",        action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import csv
import itertools

import cv2
import numpy as np
import torch
from PIL import Image

from clutter_grasp.envs.data_collection_env_cfg import DataCollectionEnvCfg
from clutter_grasp.envs.data_collection_env import DataCollectionEnv, quat_apply_batch


# ═════════════════════════════════════════════════════════════════════════════
# CSV schema — exact column order required by downstream scripts
# ═════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "image_pixels",
    "target_grasp_difficulty", "target_shape_complexity",
    "num_obstacles", "num_neighbors",
    "mean_neighbor_distance", "min_neighbor_distance",
    "mean_neighbor_grasp_difficulty", "mean_neighbor_shape_complexity",
    "free_space_volume",
    "bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height", "bbox_area",
    "obj_x", "obj_y", "obj_z", "distance_to_center", "distance_to_camera",
    "bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max",
]


# ═════════════════════════════════════════════════════════════════════════════
# EGAD id parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_egad_id(object_id: str):
    """'A1' → (1,1),  'B12' → (2,12),  'A01_2' → (1,1),  bad → (None,None)

    Same convention as randomize_scene.extract_difficulty_and_complexity: the variant
    suffix after '_' is dropped (int('01_2') would otherwise parse as 12).
    """
    try:
        s = object_id.strip().split('_')[0]
        return ord(s[0].upper()) - ord('A') + 1, int(s[1:])
    except Exception:
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# Projection — identical to collect_detection_pc_dataset.py
# ═════════════════════════════════════════════════════════════════════════════

def quat_to_rotation_matrix_ros(quat: torch.Tensor) -> torch.Tensor:
    """Exact copy from collect_detection_pc_dataset.py."""
    quat = quat.to(dtype=torch.float32)
    norm = torch.sqrt((quat ** 2).sum())
    w, x, y, z = quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm
    return torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], device=quat.device, dtype=torch.float32)


def project_points(points_world, cam_pos_w, cam_quat_w, camera_cfg):
    """
    Exact copy of project_points_to_image from collect_detection_pc_dataset.py.
    points_world : (N,3) tensor or numpy
    Returns (xs, ys) numpy arrays or None if all behind camera.
    """
    device = cam_pos_w.device
    if not isinstance(points_world, torch.Tensor):
        points_world = torch.tensor(points_world, dtype=torch.float32, device=device)
    points_world = points_world.to(device=device, dtype=torch.float32)
    cam_pos_w    = cam_pos_w.to(device=device, dtype=torch.float32)
    cam_quat_w   = cam_quat_w.to(device=device, dtype=torch.float32)

    rot_mat    = quat_to_rotation_matrix_ros(cam_quat_w)
    points_rel = points_world - cam_pos_w
    points_cam = torch.matmul(points_rel, rot_mat.T.T)   # .T.T matches working script

    valid_mask = points_cam[:, 2] > 0.01
    if valid_mask.sum() == 0:
        return None

    pts  = points_cam[valid_mask]
    w_px = camera_cfg.width
    h_px = camera_cfg.height
    f_x  = (camera_cfg.spawn.focal_length / camera_cfg.spawn.horizontal_aperture) * w_px
    c_x  = (w_px - 1) / 2.0
    c_y  = (h_px - 1) / 2.0

    xs = (f_x * (pts[:, 0] / pts[:, 2]) + c_x).cpu().numpy()
    ys = (f_x * (pts[:, 1] / pts[:, 2]) + c_y).cpu().numpy()
    return xs, ys


def project_to_bbox_px(pts_world, cam_pos_w, cam_quat_w, camera_cfg, pad=0):
    """
    Project (N,3) world points → pixel bbox (x_min, y_min, x_max, y_max) ints.
    Returns None if all behind camera.
    """
    result = project_points(pts_world, cam_pos_w, cam_quat_w, camera_cfg)
    if result is None:
        return None
    xs, ys = result
    W, H   = camera_cfg.width, camera_cfg.height
    x_min  = max(0,   int(xs.min()) - pad)
    x_max  = min(W-1, int(xs.max()) + pad)
    y_min  = max(0,   int(ys.min()) - pad)
    y_max  = min(H-1, int(ys.max()) + pad)
    return x_min, y_min, x_max, y_max


def capture_scene(env) -> tuple:
    """
    Capture RGB and read pos_w / quat_w_ros from the live camera.

    Domain randomisation (position + focal length) is handled by
    env._jitter_camera() which runs inside env.respawn_scene() before the
    settling steps and the final render.  By the time capture_scene is called
    the camera prim already has the jittered pose and env.camera.data reflects
    it.  We simply read pos_w and quat_w_ros — never compute our own rotation
    — so projection always matches what was rendered.

    Same capture sequence as collect_detection_pc_dataset.py.
    Returns (rgb_np, cam_pos_w, cam_quat_w) or (None, None, None) on failure.
    """
    if env.camera is None:
        print("  [CAM] camera is None")
        return None, None, None

    for _ in range(5):
        env.sim.step(render=True)
    env.camera.update(dt=env.cfg.sim.dt)

    try:
        rgb_data = env.camera.data.output["rgb"][0]
        rgb_np   = rgb_data.cpu().numpy()
        if rgb_np.dtype != np.uint8:
            rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
    except Exception as e:
        print(f"  [CAM] RGB capture failed: {e}")
        return None, None, None

    cam_pos_w  = env.camera.data.pos_w[0].clone()
    cam_quat_w = env.camera.data.quat_w_ros[0].clone()

    print(f"  [CAM] rgb={rgb_np.shape}  dtype={rgb_np.dtype}")
    print(f"  [CAM] pos_w ={cam_pos_w.cpu().numpy().round(3)}")
    print(f"  [CAM] quat_w={cam_quat_w.cpu().numpy().round(4)}")
    print(f"  [CAM] focal ={env.cfg.camera.spawn.focal_length:.1f} mm")

    return rgb_np, cam_pos_w, cam_quat_w


# ═════════════════════════════════════════════════════════════════════════════
# 3-D geometry — world-frame AABB using quat_apply_batch (same as working script)
# ═════════════════════════════════════════════════════════════════════════════

def get_aabb_world(env, obj_idx: int):
    """
    World-frame AABB for settled object using quat_apply_batch,
    matching extract_pointcloud_world in collect_detection_pc_dataset.py.

    Returns (pts_world, mn, mx) as float32 numpy, or None on failure.
    pts_world : (P,3) — all mesh vertices in world frame (used for projection)
    mn / mx   : (3,)  — AABB corners
    """
    try:
        obj = env.objects[obj_idx]
        obj.update(dt=env.cfg.sim.dt)

        obj_pos_w  = obj.data.root_pos_w[0].clone().to(env.device)
        obj_quat_w = obj.data.root_quat_w[0:1].clone().to(env.device)

        mesh_local = env._extract_object_mesh_for_current(obj_idx)
        if mesh_local is None:
            raise ValueError("mesh unavailable")

        if not isinstance(mesh_local, torch.Tensor):
            mesh_local = torch.tensor(mesh_local, dtype=torch.float32,
                                      device=env.device)

        pts_rot   = quat_apply_batch(obj_quat_w, mesh_local.unsqueeze(0))   # (1,P,3)
        pts_world = (pts_rot[0] + obj_pos_w).cpu().numpy().astype(np.float32)  # (P,3)

        return pts_world, pts_world.min(axis=0), pts_world.max(axis=0)

    except Exception as e:
        print(f"  [AABB] obj{obj_idx}: {e}")
        return None


def aabbs_intersect(mn1, mx1, mn2, mx2, margin):
    return np.all(mx1 + margin >= mn2) and np.all(mx2 + margin >= mn1)


def aabb_sep_distance(mn1, mx1, mn2, mx2):
    d = np.maximum(0.0, np.maximum(mn1 - mx2, mn2 - mx1))
    return float(np.linalg.norm(d))


def free_space_ratio(tgt_mn, tgt_mx, nb_aabbs, radius=0.03):
    """Fraction of expanded target AABB not occupied by any neighbour."""
    exp_mn      = tgt_mn - radius
    exp_mx      = tgt_mx + radius
    target_vol  = np.prod(np.maximum(0.0, tgt_mx - tgt_mn))
    total_space = np.prod(np.maximum(0.0, exp_mx - exp_mn))
    total_free  = total_space - target_vol
    if total_free <= 0:
        return 0.0
    free = total_free
    for (n_mn, n_mx) in nb_aabbs:
        inter_mn = np.maximum(exp_mn, n_mn)
        inter_mx = np.minimum(exp_mx, n_mx)
        free = max(0.0, free - np.prod(np.maximum(0.0, inter_mx - inter_mn)))
    return free / total_free


# ═════════════════════════════════════════════════════════════════════════════
# Per-scene metrics
# ═════════════════════════════════════════════════════════════════════════════

def collect_scene_metrics(env, target_idx: int, margin: float):
    """Compute all CSV metrics for the settled scene. Returns dict or None."""
    n_active = len(env._object_infos)
    if n_active == 0:
        return None

    aabb_data = {}
    for i in range(n_active):
        result = get_aabb_world(env, i)
        if result is not None:
            aabb_data[i] = result   # (pts_world, mn, mx)

    if target_idx not in aabb_data:
        print("  [METRICS] Target AABB unavailable")
        return None

    tgt_pts, tgt_mn, tgt_mx = aabb_data[target_idx]
    tgt_id = env._object_infos[target_idx].object_id
    tgt_gd, tgt_sc = parse_egad_id(tgt_id)
    if tgt_gd is None:
        print(f"  [METRICS] Cannot parse EGAD id '{tgt_id}'")
        return None

    nb_dists, nb_gds, nb_scs = [], [], []
    nb_aabbs     = []
    nb_aabb_data = {}

    for i, (pts, mn, mx) in aabb_data.items():
        if i == target_idx:
            continue
        if not aabbs_intersect(tgt_mn, tgt_mx, mn, mx, margin):
            continue
        nb_dists.append(aabb_sep_distance(tgt_mn, tgt_mx, mn, mx))
        nb_aabbs.append((mn, mx))
        nb_aabb_data[i] = (pts, mn, mx)
        gd, sc = parse_egad_id(env._object_infos[i].object_id)
        if gd is not None:
            nb_gds.append(gd)
            nb_scs.append(sc)

    print(f"  [METRICS] target={tgt_id}  mn={tgt_mn.round(3)}  mx={tgt_mx.round(3)}")
    print(f"  [METRICS] num_neighbors={len(nb_dists)}  "
          f"free_space={free_space_ratio(tgt_mn, tgt_mx, nb_aabbs):.3f}")

    return {
        "target_obj_id":               tgt_id,
        "target_grasp_difficulty":     tgt_gd,
        "target_shape_complexity":     tgt_sc,
        "target_pts":                  tgt_pts,
        "target_mn":                   tgt_mn,
        "target_mx":                   tgt_mx,
        "target_centre_w":             (tgt_mn + tgt_mx) / 2.0,
        "nb_aabb_data":                nb_aabb_data,
        "num_obstacles":               0,
        "num_neighbors":               len(nb_dists),
        "mean_neighbor_distance":      float(np.mean(nb_dists))  if nb_dists else 0.0,
        "min_neighbor_distance":       float(np.min(nb_dists))   if nb_dists else 0.0,
        "mean_neighbor_grasp_difficulty": float(np.mean(nb_gds)) if nb_gds  else 0.0,
        "mean_neighbor_shape_complexity": float(np.mean(nb_scs)) if nb_scs  else 0.0,
        "free_space_volume":           free_space_ratio(tgt_mn, tgt_mx, nb_aabbs),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Spatial features — analytic from world position
# ═════════════════════════════════════════════════════════════════════════════

def compute_spatial_features(target_pos_w: np.ndarray,
                              cam_pos_w_np: np.ndarray,
                              env) -> dict:
    cfg    = env.cfg
    half_w = cfg.table_width  / 2.0
    half_d = cfg.table_depth  / 2.0
    z_lo   = cfg.table_height
    z_hi   = cfg.table_height + 0.5

    tx, ty, tz = target_pos_w
    obj_x = float(np.clip((tx + half_w) / (2 * half_w), 0, 1))
    obj_y = float(np.clip((ty + half_d) / (2 * half_d), 0, 1))
    obj_z = float(np.clip((tz - z_lo)   / (z_hi - z_lo), 0, 1))

    half_diag = np.sqrt(half_w**2 + half_d**2)
    dist_c    = float(np.clip(np.sqrt(tx**2 + ty**2) / half_diag, 0, 1))
    dist_cam  = float(np.clip(np.linalg.norm(target_pos_w - cam_pos_w_np) / 2.5, 0, 1))

    return {
        "obj_x": round(obj_x, 6), "obj_y": round(obj_y, 6), "obj_z": round(obj_z, 6),
        "distance_to_center": round(dist_c,   6),
        "distance_to_camera": round(dist_cam, 6),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Drawing helpers — RGB convention throughout (cv2 calls get BGR conversion)
# ═════════════════════════════════════════════════════════════════════════════

# Colours in RGB
C_TARGET   = (0,   255,   0)    # lime   — target AABB
C_NEIGHBOUR= (255, 255,   0)    # yellow — neighbour AABBs
C_TGT_PTS  = (0,   220,   0)    # green  — target mesh dots
C_NB_PTS   = (180, 180, 180)    # grey   — neighbour mesh dots
C_SHADOW   = (0,     0,   0)
C_TEXT     = (255, 255, 255)


def _bgr(rgb): return (rgb[2], rgb[1], rgb[0])


def _rect(canvas, x0, y0, x1, y1, colour, thickness=1):
    cv2.rectangle(canvas, (x0, y0), (x1, y1), _bgr(colour), thickness)


def _dot(canvas, u, v, r, colour):
    cv2.circle(canvas, (u, v), r, _bgr(colour), -1)


def _text(canvas, txt, org, scale=0.42, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, txt, (org[0]+1, org[1]+1), font,
                scale, _bgr(C_SHADOW), thickness+1, cv2.LINE_AA)
    cv2.putText(canvas, txt, org, font,
                scale, _bgr(C_TEXT), thickness, cv2.LINE_AA)


def _draw_pts(canvas, pts_world, cam_pos_w, cam_quat_w, camera_cfg, colour, r=3):
    """Project world pts and draw dots on RGB canvas."""
    if len(pts_world) == 0:
        return 0
    result = project_points(pts_world, cam_pos_w, cam_quat_w, camera_cfg)
    if result is None:
        return 0
    H, W = canvas.shape[:2]
    xs, ys = result
    n = 0
    for x, y in zip(xs.astype(int), ys.astype(int)):
        if 0 <= x < W and 0 <= y < H:
            _dot(canvas, x, y, r, colour)
            n += 1
    return n


# ═════════════════════════════════════════════════════════════════════════════
# Render functions
# ═════════════════════════════════════════════════════════════════════════════

def render_training_image(rgb_clean, metrics, cam_pos_w, cam_quat_w, camera_cfg):
    """
    Draw lime bbox on target using projected mesh points (same as working script).
    Returns (annotated_rgb, bbox_px, bbox_norm) — bbox_norm is normalised [0,1].
    """
    W, H   = camera_cfg.width, camera_cfg.height
    canvas = rgb_clean.copy()

    bbox_px   = project_to_bbox_px(
        metrics["target_pts"], cam_pos_w, cam_quat_w, camera_cfg, pad=4
    )
    print(f"  [RENDER] target bbox_px={bbox_px}  image={W}x{H}")

    bbox_norm = None
    if bbox_px is not None:
        x0, y0, x1, y1 = bbox_px
        _rect(canvas, x0, y0, x1, y1, C_TARGET, thickness=2)
        w_n = (x1 - x0) / W
        h_n = (y1 - y0) / H
        bbox_norm = {
            "bbox_x_min":    x0 / W,   "bbox_y_min":    y0 / H,
            "bbox_x_max":    x1 / W,   "bbox_y_max":    y1 / H,
            "bbox_center_x": ((x0+x1)/2) / W,
            "bbox_center_y": ((y0+y1)/2) / H,
            "bbox_width":    w_n, "bbox_height": h_n, "bbox_area": w_n * h_n,
        }
    else:
        print("  [RENDER] WARNING: target not visible — no bbox drawn")

    return canvas, bbox_px, bbox_norm


def render_validation_image(rgb_clean, metrics, cam_pos_w, cam_quat_w,
                             camera_cfg, env):
    """
    Full metric overlay.  Draws (back→front):
      grey  dots  : neighbour mesh points
      green dots  : target mesh points
      yellow boxes: neighbour AABBs (labelled)
      lime  box   : target AABB
      panel       : metric values
      legend      : colour key
    """
    W, H   = camera_cfg.width, camera_cfg.height
    canvas = rgb_clean.copy()

    print(f"  [VAL] cam_pos ={cam_pos_w.cpu().numpy().round(3)}")
    print(f"  [VAL] cam_quat={cam_quat_w.cpu().numpy().round(4)}")

    # Neighbour mesh dots
    for i, (pts, mn, mx) in metrics["nb_aabb_data"].items():
        n = _draw_pts(canvas, pts, cam_pos_w, cam_quat_w, camera_cfg,
                      C_NB_PTS, r=2)
        print(f"  [VAL] neighbour {env._object_infos[i].object_id}: "
              f"{n}/{len(pts)} pts drawn")

    # Target mesh dots
    n = _draw_pts(canvas, metrics["target_pts"],
                  cam_pos_w, cam_quat_w, camera_cfg, C_TGT_PTS, r=3)
    print(f"  [VAL] target mesh pts drawn: {n}/{len(metrics['target_pts'])}")

    # Neighbour AABB boxes (project all 8 corners of each neighbour AABB)
    for i, (pts, mn, mx) in metrics["nb_aabb_data"].items():
        # Use the actual mesh pts for neighbour bbox too — same as training img
        nb_bbox = project_to_bbox_px(pts, cam_pos_w, cam_quat_w, camera_cfg)
        if nb_bbox is None:
            continue
        x0, y0, x1, y1 = nb_bbox
        _rect(canvas, x0, y0, x1, y1, C_NEIGHBOUR, thickness=1)
        dist  = aabb_sep_distance(metrics["target_mn"], metrics["target_mx"], mn, mx)
        obj_id = env._object_infos[i].object_id
        _text(canvas, f"{obj_id} d={dist:.3f}m",
              (x0+2, min(y0+14, H-4)), scale=0.36)

    # Target AABB box
    tgt_bbox = project_to_bbox_px(
        metrics["target_pts"], cam_pos_w, cam_quat_w, camera_cfg, pad=4
    )
    if tgt_bbox is not None:
        x0, y0, x1, y1 = tgt_bbox
        _rect(canvas, x0, y0, x1, y1, C_TARGET, thickness=3)
        _text(canvas,
              f"TARGET {metrics['target_obj_id']}  "
              f"GD={metrics['target_grasp_difficulty']}  "
              f"SC={metrics['target_shape_complexity']}",
              (x0+2, max(y0-6, 14)), scale=0.40)

    # Metrics panel
    panel = [
        f"num_neighbors  : {metrics['num_neighbors']}",
        f"free_space_vol : {metrics['free_space_volume']:.4f}",
        f"mean_nb_dist   : {metrics['mean_neighbor_distance']:.4f} m",
        f"min_nb_dist    : {metrics['min_neighbor_distance']:.4f} m",
        f"mean_nb_GD     : {metrics['mean_neighbor_grasp_difficulty']:.2f}",
        f"mean_nb_SC     : {metrics['mean_neighbor_shape_complexity']:.2f}",
    ]
    px, py = 8, 18
    for line in panel:
        _text(canvas, line, (px, py), scale=0.42)
        py += 17

    # Legend
    legend = [
        (C_TARGET,    "target bbox (lime)"),
        (C_NEIGHBOUR, "neighbour bboxes (yellow)"),
        (C_TGT_PTS,   "target mesh pts (green)"),
        (C_NB_PTS,    "neighbour mesh pts (grey)"),
    ]
    lx, ly = 8, H - len(legend)*17 - 6
    for colour, label in legend:
        cv2.rectangle(canvas, (lx, ly-9), (lx+12, ly+3), _bgr(colour), -1)
        _text(canvas, label, (lx+16, ly), scale=0.36)
        ly += 17

    return canvas



# ═════════════════════════════════════════════════════════════════════════════
# Image-level domain randomisation
#
# TiledCamera bakes its viewport at sensor creation — moving the USD prim at
# runtime produces a blank image.  Instead we randomise at the image level:
#   brightness : ×Uniform(0.7, 1.3)
#   contrast   : blend toward grey ×Uniform(0.8, 1.2)
#   saturation : scale HSV S channel ×Uniform(0.7, 1.3)
#   h_flip     : 50 % chance of horizontal flip
# These transformations are applied to the saved training image only —
# NOT to rgb_clean, so the validation image always shows the raw scene and
# the bbox / metric computations are unaffected.
# ═════════════════════════════════════════════════════════════════════════════

def augment_image(rgb: np.ndarray) -> np.ndarray:
    """
    Apply randomised photometric augmentation to an RGB uint8 image.
    Returns a new uint8 array — input is not modified.
    """
    img = rgb.astype(np.float32) / 255.0

    # Brightness
    img = img * np.random.uniform(0.7, 1.3)

    # Contrast — blend toward per-image mean grey
    mean = img.mean()
    img  = mean + (img - mean) * np.random.uniform(0.8, 1.2)

    img = np.clip(img, 0.0, 1.0)

    # Saturation — in HSV space
    hsv        = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * np.random.uniform(0.7, 1.3), 0, 255)
    img        = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Horizontal flip (bbox is also flipped so CSV values stay consistent)
    flipped = np.random.rand() < 0.5

    out = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    if flipped:
        out = out[:, ::-1, :].copy()

    return out, flipped

# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    out_dir  = Path(args_cli.output_dir).resolve()
    img_dir  = out_dir / "images"
    val_dir  = out_dir / "validation"
    csv_path = out_dir / "classifier_data.csv"

    img_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.validate:
        val_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = DataCollectionEnvCfg()
    env_cfg.scene.num_envs       = 1
    env_cfg.min_objects_to_spawn = args_cli.min_objects
    env_cfg.max_objects_to_spawn = args_cli.max_objects
    env_cfg.spawn_settling_steps = args_cli.settling_steps
    env_cfg.object_usd_dir       = str(Path(args_cli.usd_dir).resolve())

    print(f"\n[COLLECT] Output dir      : {out_dir}")
    print(f"[COLLECT] USD dir         : {env_cfg.object_usd_dir}")
    print(f"[COLLECT] Target scenes   : {args_cli.num_scenes}")
    print(f"[COLLECT] Objects range   : {args_cli.min_objects}–{args_cli.max_objects}")
    print(f"[COLLECT] Neighbour margin: {args_cli.neighbor_margin} m")
    print(f"[COLLECT] Validation      : {'ON' if args_cli.validate else 'OFF'}\n")

    try:
        env = DataCollectionEnv(env_cfg)
    except Exception as e:
        print(f"[ERROR] Env creation failed: {e}")
        simulation_app.close()
        return

    csv_exists = csv_path.exists()
    csv_file   = open(csv_path, "a", newline="")
    writer     = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    scenes_saved  = 0
    scenes_failed = 0

    while scenes_saved < args_cli.num_scenes and simulation_app.is_running():
        print(f"\n{'='*60}")
        print(f"[COLLECT] Scene {scenes_saved+1} / {args_cli.num_scenes}")
        print(f"{'='*60}")

        if scenes_saved > 0:
            try:
                env.respawn_scene()
            except Exception as e:
                print(f"[WARN] Respawn: {e}")
                scenes_failed += 1
                continue

        n_active = len(env._object_infos)
        if n_active == 0:
            print("[WARN] No active objects")
            scenes_failed += 1
            continue

        # Pick random target
        target_idx = int(np.random.randint(0, n_active))
        target_id  = env._object_infos[target_idx].object_id
        print(f"[COLLECT] Target slot {target_idx} = '{target_id}'")

        # Z-height guard
        env.objects[target_idx].update(dt=env.cfg.sim.dt)
        tz = env.objects[target_idx].data.root_pos_w[0, 2].item()
        if tz < env.cfg.table_height:
            print(f"[WARN] Target z={tz:.3f} < table={env.cfg.table_height:.3f}")
            scenes_failed += 1
            continue

        # ── CAPTURE FIRST (jittered camera + sim step) ────────────────────
        # Reading pos_w / quat_w_ros AFTER stepping ensures projection
        # matches the rendered image exactly.
        rgb_clean, cam_pos_w, cam_quat_w = capture_scene(env)
        if rgb_clean is None:
            print("[WARN] Capture failed")
            scenes_failed += 1
            continue

        # ── METRICS (same physics state as image) ─────────────────────────
        metrics = collect_scene_metrics(env, target_idx, args_cli.neighbor_margin)
        if metrics is None:
            print("[WARN] Metric extraction failed")
            scenes_failed += 1
            continue

        # ── TRAINING IMAGE ────────────────────────────────────────────────
        scene_stem = f"scene_{scenes_saved:05d}"
        image_path = img_dir / f"{scene_stem}.png"

        rgb_train, bbox_px, bbox_norm = render_training_image(
            rgb_clean, metrics, cam_pos_w, cam_quat_w, env.cfg.camera
        )

        # Apply photometric augmentation to training image only
        # (validation images always use the raw unaugmented scene)
        rgb_aug, was_flipped = augment_image(rgb_train)

        # If horizontally flipped, mirror bbox x-coordinates
        if was_flipped and bbox_norm is not None:
            W = env.cfg.camera.width
            # Pixel coords
            x0 = int(bbox_norm["bbox_x_min"] * W)
            x1 = int(bbox_norm["bbox_x_max"] * W)
            new_x0 = W - 1 - x1
            new_x1 = W - 1 - x0
            w_n = bbox_norm["bbox_width"]
            bbox_norm["bbox_x_min"]    = new_x0 / W
            bbox_norm["bbox_x_max"]    = new_x1 / W
            bbox_norm["bbox_center_x"] = ((new_x0 + new_x1) / 2) / W
            # width, height, area unchanged by horizontal flip

        Image.fromarray(rgb_aug, mode='RGB').save(str(image_path))

        # ── VALIDATION IMAGES ─────────────────────────────────────────────
        if args_cli.validate:
            Image.fromarray(rgb_clean, mode='RGB').save(
                str(val_dir / f"{scene_stem}_clean.png")
            )
            val_img = render_validation_image(
                rgb_clean, metrics, cam_pos_w, cam_quat_w, env.cfg.camera, env
            )
            Image.fromarray(val_img, mode='RGB').save(
                str(val_dir / f"{scene_stem}_metrics.png")
            )

        # ── BBOX NORM FALLBACK ────────────────────────────────────────────
        _nan = float("nan")
        if bbox_norm is None:
            bbox_norm = {k: _nan for k in [
                "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
                "bbox_center_x", "bbox_center_y",
                "bbox_width", "bbox_height", "bbox_area",
            ]}

        # ── SPATIAL FEATURES ──────────────────────────────────────────────
        cam_pos_np = cam_pos_w.cpu().numpy().astype(np.float32)
        spatial    = compute_spatial_features(
            metrics["target_centre_w"], cam_pos_np, env
        )

        # ── WRITE CSV ─────────────────────────────────────────────────────
        def _r(v, n=6):
            return round(float(v), n) if not (
                isinstance(v, float) and np.isnan(v)) else _nan

        row = {
            "image_pixels":                   str(image_path),
            "target_grasp_difficulty":        metrics["target_grasp_difficulty"],
            "target_shape_complexity":        metrics["target_shape_complexity"],
            "num_obstacles":                  0,
            "num_neighbors":                  metrics["num_neighbors"],
            "mean_neighbor_distance":         _r(metrics["mean_neighbor_distance"]),
            "min_neighbor_distance":          _r(metrics["min_neighbor_distance"]),
            "mean_neighbor_grasp_difficulty": _r(metrics["mean_neighbor_grasp_difficulty"], 4),
            "mean_neighbor_shape_complexity": _r(metrics["mean_neighbor_shape_complexity"], 4),
            "free_space_volume":              _r(metrics["free_space_volume"]),
            "bbox_center_x": _r(bbox_norm["bbox_center_x"]),
            "bbox_center_y": _r(bbox_norm["bbox_center_y"]),
            "bbox_width":    _r(bbox_norm["bbox_width"]),
            "bbox_height":   _r(bbox_norm["bbox_height"]),
            "bbox_area":     _r(bbox_norm["bbox_area"], 8),
            "obj_x":               spatial["obj_x"],
            "obj_y":               spatial["obj_y"],
            "obj_z":               spatial["obj_z"],
            "distance_to_center":  spatial["distance_to_center"],
            "distance_to_camera":  spatial["distance_to_camera"],
            "bbox_x_min":    _r(bbox_norm["bbox_x_min"]),
            "bbox_x_max":    _r(bbox_norm["bbox_x_max"]),
            "bbox_y_min":    _r(bbox_norm["bbox_y_min"]),
            "bbox_y_max":    _r(bbox_norm["bbox_y_max"]),
        }
        writer.writerow(row)
        csv_file.flush()

        scenes_saved += 1
        print(
            f"[COLLECT] ✓ {scene_stem} | {target_id} "
            f"GD={metrics['target_grasp_difficulty']} SC={metrics['target_shape_complexity']} "
            f"| nb={metrics['num_neighbors']} fsv={metrics['free_space_volume']:.3f} "
            f"| bbox_px={bbox_px}"
        )

        if scenes_saved % 50 == 0:
            _summary(scenes_saved, scenes_failed, args_cli.num_scenes, csv_path)

    csv_file.close()
    env.close()
    _summary(scenes_saved, scenes_failed, args_cli.num_scenes, csv_path)


def _summary(saved, failed, target, csv_path):
    print(f"\n{'='*60}")
    print(f"[COLLECT] SUMMARY  saved={saved}/{target}  failed={failed}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[COLLECT] Interrupted")
    except Exception as e:
        import traceback
        print(f"\n[COLLECT] Fatal error: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()