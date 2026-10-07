#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the clutter grasping protocol for one target object with the Contactile hand.

One run = one object in one condition (isolated, C0_easy, C1_medium or C2_hard), repeated for
--num_trials trials. Cluttered scenes are generated from the protocol's clutter levels and
confirmed by the clutter classifier before the trials start (and again at every respawn).

Controller (--policy, names as in the paper and analysis/):
  rl           per-object PPO policies trained in isolation (reference)   weights/ppo_policies/
  rl_clutter   per-object PPO policies trained in clutter                weights/ppo_clutter_policies/
  transformer  one transformer distilled from the rl policies            weights/transformer/distillation/
  distilled    one transformer distilled from the rl_clutter policies    weights/transformer/cluttered_distillation/
  --controller my_pkg.my_module:make_controller   your own controller (see README.md)

Results: results/benchmark/<controller>/single_<object>[_<condition>]_<timestamp>/results/results.json
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Repository layout (see README.md): this file lives in scripts/benchmark/.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
WEIGHTS_DIR = REPO_ROOT / "weights"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the clutter grasping protocol for one target object.")
parser.add_argument("--target_object", type=str, required=True,
                   help="Object to pick (USD file stem, e.g. 'A24_0')")
parser.add_argument("--target_complexity", type=str, default="",
                   choices=["C0_easy", "C1_medium", "C2_hard"],
                   help="Clutter level (scene is confirmed by the clutter classifier)")
parser.add_argument("--isolated", action="store_true",
                   help="Isolated condition: target object only, no clutter")
parser.add_argument("--num_clutter", type=int, default=10,
                   help="Number of randomly placed clutter objects when neither --isolated nor --target_complexity is given")
parser.add_argument("--num_trials", type=int, default=5,
                   help="Number of trials to run")
parser.add_argument("--max_steps_per_trial", type=int, default=200,
                   help="Max steps per trial before hand respawn")
parser.add_argument("--max_attempts", type=int, default=100,
                   help="Max drop attempts before giving up")
POLICIES = {
    # name: (per-object PPO dir or None, transformer checkpoint dir or None)
    "rl": (WEIGHTS_DIR / "ppo_policies", None),
    "rl_clutter": (WEIGHTS_DIR / "ppo_clutter_policies", None),
    "transformer": (None, WEIGHTS_DIR / "transformer" / "distillation"),
    "distilled": (None, WEIGHTS_DIR / "transformer" / "cluttered_distillation"),
}
parser.add_argument("--policy", type=str, default="rl", choices=list(POLICIES),
                   help="Released controller to run (default: rl). Ignored with --controller.")
parser.add_argument("--trained_policies_dir", type=str, default=None,
                   help="Per-object PPO policies: <dir>/<object>/model.zip (+ model_vecnormalize.pkl). "
                        "Default: the --policy's folder.")
parser.add_argument("--classifier_model_dir", type=str, default=str(WEIGHTS_DIR / "classifier"),
                   help="Clutter classifier (best_model.pth + config.json)")
parser.add_argument("--controller", type=str, default="",
                   help="Your own controller as 'module.path:factory' (default: per-object PPO). "
                        "factory(env) must return an object with act(obs) -> actions.")
parser.add_argument("--controller_name", type=str, default="",
                   help="Name used for the results folder (default: 'ppo' or the factory name)")
# ── Distilled transformer (same flags as the original runner; --policy transformer/distilled set them) ──
parser.add_argument("--use_transformer", action="store_true",
                   help="Use a distilled transformer instead of per-object PPO")
parser.add_argument("--transformer_checkpoint", type=str, default=None,
                   help="best_model.pth from distillation (default: the --policy's checkpoint)")
parser.add_argument("--transformer_obs_norm_stats", type=str, default=None,
                   help="obs_norm_stats.npz from distillation (default: next to the checkpoint)")
parser.add_argument("--transformer_d_model",            type=int, default=None)
parser.add_argument("--transformer_nhead",              type=int, default=None)
parser.add_argument("--transformer_num_layers",         type=int, default=None)
parser.add_argument("--transformer_dim_feedforward",    type=int, default=None)
parser.add_argument("--transformer_num_proprio_tokens", type=int, default=None,
                   help="Architecture overrides; default: config.json next to the checkpoint")
parser.add_argument("--video", action="store_true",
                   help="Record video")
parser.add_argument("--video_length", type=int, default=100000,
                   help="Video length in steps")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Resolve the controller before the simulator starts, so bad paths fail fast.
_ppo_dir, _tf_dir = POLICIES[args_cli.policy]
if _tf_dir is not None or args_cli.transformer_checkpoint or args_cli.transformer_obs_norm_stats:
    args_cli.use_transformer = True
if args_cli.use_transformer and _tf_dir is None:
    _tf_dir = POLICIES["transformer"][1]
    if args_cli.policy == "rl":
        args_cli.policy = "transformer"
if args_cli.controller:
    args_cli.use_transformer = False
elif args_cli.use_transformer:
    args_cli.transformer_checkpoint = args_cli.transformer_checkpoint or str(_tf_dir / "best_model.pth")
    for f in [args_cli.transformer_checkpoint, args_cli.transformer_obs_norm_stats]:
        if f and not Path(f).exists():
            parser.error(f"{f} not found (run scripts/download_weights.sh transformer)")
else:
    args_cli.trained_policies_dir = args_cli.trained_policies_dir or str(_ppo_dir)
    if not Path(args_cli.trained_policies_dir).is_dir():
        parser.error(f"{args_cli.trained_policies_dir} not found (run scripts/download_weights.sh)")
if args_cli.trained_policies_dir is None:
    args_cli.trained_policies_dir = str(POLICIES["rl"][0])  # only used to list objects

# Cameras are needed for video and for the clutter classifier.
if args_cli.video or args_cli.target_complexity:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after simulator launch"""

import gymnasium as gym
import time
import torch
from datetime import datetime
import cv2
import json
import pickle

from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import clutter_grasp.envs  # noqa: F401  (registers the gym environments)
from clutter_grasp.envs.benchmark_env_cfg import BenchmarkEnvCfg
from clutter_grasp.controllers.transformer_controller import TransformerController
from clutter_grasp.protocol.clutter_levels import (
    CLUTTER_CONFIGS,
    NEIGHBOR_RADIUS as RADIUS,
    apply_complexity_correction as _apply_complexity_correction,
    count_neighbors as _count_neighbors_from_features,
)
from clutter_grasp.protocol.classifier import (
    estimate_3d_position_from_bbox,
    load_classifier_model,
    predict_complexity_from_image,
)


# ==============================================
# CONTROLLERS
# ==============================================

class PPOController:
    """Per-object PPO policy (the paper's reference controller), loaded by the environment."""

    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        with torch.no_grad():
            actions, _ = self.policy.predict(obs, deterministic=True)
        return actions


def load_external_controller(spec: str, env):
    """Import 'module.path:factory' and build the controller with factory(env)."""
    module_name, _, factory_name = spec.partition(":")
    if not factory_name:
        raise ValueError(f"--controller must look like 'module.path:factory', got '{spec}'")
    factory = getattr(importlib.import_module(module_name), factory_name)
    controller = factory(env)
    if not hasattr(controller, "act"):
        raise TypeError(f"{spec} returned {type(controller).__name__}, which has no act(obs) method")
    return controller


# ==============================================
# CLUTTER-LEVEL VERIFICATION (classifier + neighbour count)
# ==============================================

def verify_target_object_complexity(env, target_complexity: str, classifier_model,
                                    args, max_attempts: int = 20) -> tuple:
    """Verify that target object (index 0) matches desired complexity."""
    complexity_map = {'C0_easy': 0, 'C1_medium': 1, 'C2_hard': 2}
    desired_level = complexity_map[target_complexity]

    print(f"\n{'='*80}")
    print(f"[VERIFY] Verifying target object complexity")
    print(f"[VERIFY] Desired: {target_complexity} (level {desired_level})")
    print(f"{'='*80}")

    verified_image_path = None

    for attempt in range(max_attempts):
        print(f"\n[VERIFY] Attempt {attempt + 1}/{max_attempts}")

        # Move hand away for clean image
        safe_pos = env.scene.env_origins.clone()
        safe_pos[:, 0] = 10.0
        safe_pos[:, 1] = 10.0
        safe_pos[:, 2] = -5.0

        hand_state = env.robot.data.default_root_state.clone()
        hand_state[:, 0:3] = safe_pos
        hand_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        hand_state[:, 7:] = 0.0
        env.robot.write_root_state_to_sim(hand_state)

        all_joint_pos = env.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(all_joint_pos)
        env.robot.write_joint_state_to_sim(all_joint_pos, joint_vel, None)

        print(f"[VERIFY] Hand moved away, settling scene...")

        for i in range(50):
            env.sim.step(render=True)
            if i % 10 == 0:
                for obj in env.objects:
                    obj.update(dt=env.cfg.sim.dt)
                if env.classifier_camera is not None:
                    env.classifier_camera.update(dt=env.cfg.sim.dt)

        for obj in env.objects:
            obj.update(dt=env.cfg.sim.dt)
        if env.classifier_camera is not None:
            env.classifier_camera.update(dt=env.cfg.sim.dt)

        # Capture base image
        if env.classifier_camera is None:
            print(f"[ERROR] Classifier camera is None!")
            return False, None

        try:
            env.classifier_camera.update(dt=env.cfg.sim.dt)

            if "rgb" not in env.classifier_camera.data.output:
                print(f"[ERROR] 'rgb' not in camera output")
                continue

            rgb_data   = env.classifier_camera.data.output["rgb"][0]
            rgb_np     = rgb_data.cpu().numpy()
            rgb_uint8  = (rgb_np * 255).astype(np.uint8) if rgb_np.dtype != np.uint8 else rgb_np
            base_image_np = rgb_uint8
            print(f"[VERIFY] ✓ Base image captured")

        except Exception as e:
            print(f"[ERROR] Failed to capture image: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Extract spatial features for ALL objects
        camera_data = env.classifier_camera.data
        cam_pos  = camera_data.pos_w[0].clone()
        cam_quat = camera_data.quat_w_ros[0].clone()

        all_bbox_features    = []
        all_spatial_features = []

        width  = env.cfg.classifier_camera.width
        height = env.cfg.classifier_camera.height

        for obj_idx in range(len(env.objects)):
            obj_info = env._object_infos[obj_idx]
            mesh_points_world = extract_policy_pointcloud_for_object(env, obj_idx, obj_info)

            if mesh_points_world is not None and len(mesh_points_world) > 0:
                projected_2d = project_points_to_image_cached(
                    mesh_points_world, cam_pos, cam_quat, env.cfg.classifier_camera
                )

                if projected_2d is not None:
                    xs, ys = projected_2d
                    padding = 2
                    x_min = max(0, int(xs.min()) - padding)
                    x_max = min(width - 1, int(xs.max()) + padding)
                    y_min = max(0, int(ys.min()) - padding)
                    y_max = min(height - 1, int(ys.max()) + padding)

                    bbox_features = {
                        'bbox_x_min':    x_min / width,
                        'bbox_x_max':    x_max / width,
                        'bbox_y_min':    y_min / height,
                        'bbox_y_max':    y_max / height,
                        'bbox_center_x': ((x_min + x_max) / 2) / width,
                        'bbox_center_y': ((y_min + y_max) / 2) / height,
                        'bbox_width':    (x_max - x_min) / width,
                        'bbox_height':   (y_max - y_min) / height,
                        'bbox_area':     ((x_max - x_min) * (y_max - y_min)) / (width * height),
                    }

                    position_features = estimate_3d_position_from_bbox(bbox_features)
                    spatial_features  = np.array([
                        position_features['obj_x'],
                        position_features['obj_y'],
                        position_features['distance_to_center'],
                        bbox_features['bbox_area'],
                        bbox_features['bbox_width'],
                        bbox_features['bbox_height'],
                    ], dtype=np.float32)

                    all_bbox_features.append(bbox_features)
                    all_spatial_features.append(spatial_features)

                    if obj_idx == 0:
                        print(f"[VERIFY] Target bbox: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                else:
                    all_bbox_features.append(None)
                    all_spatial_features.append(None)
            else:
                all_bbox_features.append(None)
                all_spatial_features.append(None)

        if all_spatial_features[0] is None:
            print(f"[ERROR] No features for target, retrying spawn...")
            _respawn_scene_with_new_layout(env)
            continue

        target_neighbor_count = _count_neighbors_from_features(
            0, all_spatial_features, radius=RADIUS
        )
        print(f"[VERIFY] Target has {target_neighbor_count} neighbors (radius={RADIUS}m)")

        # Temp annotated image for classifier
        temp_annotated = base_image_np.copy()
        if all_bbox_features[0] is not None:
            bbox  = all_bbox_features[0]
            x_min = int(bbox['bbox_x_min'] * width)
            x_max = int(bbox['bbox_x_max'] * width)
            y_min = int(bbox['bbox_y_min'] * height)
            y_max = int(bbox['bbox_y_max'] * height)
            cv2.rectangle(temp_annotated, (x_min, y_min), (x_max, y_max), (50, 255, 50), 2)

        temp_image_path = Path(env.cfg.classifier_image_dir) / f"temp_target_attempt_{attempt}.png"
        Image.fromarray(temp_annotated).save(str(temp_image_path))

        device_cls = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_complexity = predict_complexity_from_image(
            classifier_model, str(temp_image_path), device_cls, verbose=True
        )

        print(f"\n[VERIFY] Classification Results:")
        print(f"  Object: {env._object_infos[0].object_id}")
        print(f"  Raw Prediction: C{predicted_complexity}")
        print(f"  Neighbor Count: {target_neighbor_count}")
        print(f"  Desired: C{desired_level}")

        corrected_complexity = _apply_complexity_correction(
            predicted_complexity, target_neighbor_count,
            obj_name=env._object_infos[0].object_id
        )
        print(f"  Corrected: C{corrected_complexity}")

        temp_image_path.unlink()

        if corrected_complexity == desired_level:
            print(f"\n[VERIFY] ✓✓✓ SUCCESS! Target complexity matches desired level")
            print(f"[VERIFY] Verification took {attempt + 1} attempt(s)")

            # Build final annotated image
            final_annotated = base_image_np.copy()

            if all_bbox_features[0] is not None:
                bbox  = all_bbox_features[0]
                x_min = int(bbox['bbox_x_min'] * width)
                x_max = int(bbox['bbox_x_max'] * width)
                y_min = int(bbox['bbox_y_min'] * height)
                y_max = int(bbox['bbox_y_max'] * height)
                cv2.rectangle(final_annotated, (x_min, y_min), (x_max, y_max), (50, 255, 50), 3)

            neighbors_drawn = 0
            non_neighbors_drawn = 0

            for obj_idx in range(1, len(all_spatial_features)):
                if all_spatial_features[obj_idx] is None:
                    continue
                obj_x    = all_spatial_features[obj_idx][0]
                obj_y    = all_spatial_features[obj_idx][1]
                target_x = all_spatial_features[0][0]
                target_y = all_spatial_features[0][1]
                distance = np.sqrt((obj_x - target_x)**2 + (obj_y - target_y)**2)

                bbox = all_bbox_features[obj_idx]
                if bbox is not None:
                    x_min = int(bbox['bbox_x_min'] * width)
                    x_max = int(bbox['bbox_x_max'] * width)
                    y_min = int(bbox['bbox_y_min'] * height)
                    y_max = int(bbox['bbox_y_max'] * height)

                    if distance < RADIUS:
                        cv2.rectangle(final_annotated, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                        neighbors_drawn += 1
                    else:
                        cv2.rectangle(final_annotated, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
                        non_neighbors_drawn += 1

            # Labels
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thick = 2
            y_offset   = 30

            for text in [
                f"Target: {env._object_infos[0].object_id}",
                f"Complexity: C{corrected_complexity} ({target_complexity})",
                f"Neighbors: {target_neighbor_count}",
                f"Non-neighbors: {non_neighbors_drawn}",
            ]:
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
                cv2.rectangle(final_annotated, (10, y_offset - th - 5), (20 + tw, y_offset + 5), (255, 255, 255), -1)
                cv2.putText(final_annotated, text, (15, y_offset), font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)
                y_offset += th + 15

            # Legend
            legend_x = width - 200
            legend_y = height - 120
            cv2.rectangle(final_annotated, (legend_x - 10, legend_y - 10), (width - 10, height - 10), (255, 255, 255), -1)
            y_pos = legend_y + 10
            for label_text, color in [("Target", (50, 255, 50)), ("Neighbor", (0, 255, 255)), ("Non-neighbor", (0, 165, 255))]:
                cv2.rectangle(final_annotated, (legend_x, y_pos - 10), (legend_x + 20, y_pos + 5), color, -1)
                cv2.putText(final_annotated, label_text, (legend_x + 30, y_pos), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                y_pos += 25

            verified_image_path = Path(env.cfg.classifier_image_dir) / f"verified_{target_complexity}.png"
            Image.fromarray(final_annotated).save(str(verified_image_path))

            print(f"[VERIFY] ✓ Saved verified image: {verified_image_path.name}")
            print(f"{'='*80}\n")

            return True, str(verified_image_path)
        else:
            print(f"\n[VERIFY] ✗ Mismatch: C{corrected_complexity} != C{desired_level}")
            print(f"[VERIFY] Respawning scene...")
            _respawn_scene_with_new_layout(env)

    print(f"\n[VERIFY] ✗✗✗ Failed after {max_attempts} attempts")
    print(f"{'='*80}\n")
    return False, None


def _respawn_scene_with_new_layout(env):
    """Respawn all objects with a new random layout."""
    print(f"[RESPAWN] Generating new scene layout...")
    env._initialize_cluttered_scene()
    if hasattr(env.cfg, 'scene_random_seed'):
        env.cfg.scene_random_seed += 1
    print(f"[RESPAWN] ✓ New scene layout generated")


# ==============================================
# SCENE / RESULT HELPERS
# ==============================================

def extract_policy_pointcloud_for_object(env, obj_idx: int, obj_info):
    """Extract point cloud for object."""
    from clutter_grasp.envs.benchmark_env import quat_apply_batch

    try:
        for obj in env.objects:
            obj.update(dt=env.cfg.sim.dt)

        for i in range(20):
            env.sim.step(render=False)
            if i % 5 == 0:
                for obj in env.objects:
                    obj.update(dt=env.cfg.sim.dt)

        for obj in env.objects:
            obj.update(dt=env.cfg.sim.dt)

        mesh_points_local = env._extract_object_mesh_for_current(obj_idx)

        if mesh_points_local is None:
            return None

        if not isinstance(mesh_points_local, torch.Tensor):
            mesh_points_local = torch.tensor(mesh_points_local, device=env.device, dtype=torch.float32)

        current_object = env.objects[obj_idx]
        current_object.update(dt=env.cfg.sim.dt)

        obj_pos_world  = current_object.data.root_pos_w[0].clone()
        obj_quat_world = current_object.data.root_quat_w[0:1].clone()

        local_points = mesh_points_local.unsqueeze(0)
        world_points = quat_apply_batch(obj_quat_world, local_points)
        world_points = world_points + obj_pos_world.unsqueeze(0).unsqueeze(1)
        world_points = world_points[0]

        return world_points

    except Exception as e:
        print(f"  [ERROR] Point cloud extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def project_points_to_image_cached(points_world, cam_pos_w, cam_quat_w, camera_cfg):
    """Project 3D world points to 2D image coordinates."""
    cam_pos_w  = cam_pos_w.to(dtype=torch.float32)
    cam_quat_w = cam_quat_w.to(dtype=torch.float32)

    rot_mat      = quat_to_rotation_matrix_ros(cam_quat_w)
    points_rel   = points_world - cam_pos_w
    points_cam   = torch.matmul(points_rel, rot_mat.T.T)

    valid_mask = points_cam[:, 2] > 0.01
    if valid_mask.sum() == 0:
        return None

    points_cam_valid = points_cam[valid_mask]
    width  = camera_cfg.width
    height = camera_cfg.height

    focal_length_mm = camera_cfg.spawn.focal_length
    h_aperture_mm   = camera_cfg.spawn.horizontal_aperture
    f_x = (focal_length_mm / h_aperture_mm) * width
    f_y = f_x
    c_x = (width  - 1) / 2.0
    c_y = (height - 1) / 2.0

    xs = f_x * (points_cam_valid[:, 0] / points_cam_valid[:, 2]) + c_x
    ys = f_y * (points_cam_valid[:, 1] / points_cam_valid[:, 2]) + c_y

    return xs.cpu().numpy(), ys.cpu().numpy()


def quat_to_rotation_matrix_ros(quat):
    """Convert ROS quaternion to rotation matrix."""
    quat      = quat.to(dtype=torch.float32)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    norm      = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], device=quat.device, dtype=torch.float32)


def calculate_scene_chaos(initial_positions, final_positions, object_status, table_bounds=None):
    """Calculate normalized movement of target object."""
    table_size     = 0.85
    x_min, x_max   = -table_size / 2, table_size / 2
    y_min, y_max   = -table_size / 2, table_size / 2
    table_diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)

    if table_bounds is not None:
        x_min, x_max, y_min, y_max = table_bounds
        table_diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)

    empty = {
        'target_distance': 0.0, 'target_normalized_distance': 0.0,
        'target_dx': 0.0, 'target_dy': 0.0, 'target_status': 'no_data',
        'table_diagonal': float(table_diagonal), 'initial_pos': None, 'final_pos': None,
    }

    if initial_positions is None or final_positions is None:
        return empty
    if 0 not in initial_positions or 0 not in final_positions:
        return empty

    status      = object_status.get(0, 'unknown')
    initial_pos = initial_positions[0]
    final_pos   = final_positions[0]

    dx       = final_pos[0] - initial_pos[0]
    dy       = final_pos[1] - initial_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    norm_d   = distance / table_diagonal if table_diagonal > 0 else 0.0

    return {
        'target_distance':            float(distance),
        'target_normalized_distance': float(norm_d),
        'target_dx':                  float(dx),
        'target_dy':                  float(dy),
        'target_status':              status,
        'table_diagonal':             float(table_diagonal),
        'initial_pos':                initial_positions.copy(),
        'final_pos':                  final_positions.copy(),
    }


# ==============================================
# RESULTS SAVING
# ==============================================

def save_simple_results(exp_folder: Path, results: dict):
    """Save results including all trials."""
    results_dir = exp_folder / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Results saved to: {json_path}")

    if 'trial_results' in results and results['trial_results']:
        trials_path = results_dir / "trial_results.json"
        with open(trials_path, 'w') as f:
            json.dump(results['trial_results'], f, indent=2)
        print(f"[SAVE] Trial-by-trial results saved to: {trials_path}")

        trial_list = results['trial_results']
        print(f"\n{'='*80}")
        print(f"TRIAL SUMMARY")
        print(f"{'='*80}")
        successes = sum(1 for t in trial_list if t['success'])
        print(f"Successful trials: {successes}/{len(trial_list)}")
        print(f"Average drops:         {np.mean([t['drops'] for t in trial_list]):.2f}")
        print(f"Average hand respawns: {np.mean([t['hand_respawns'] for t in trial_list]):.2f}")
        print(f"Average steps:         {np.mean([t['steps'] for t in trial_list]):.0f}")
        print(f"{'='*80}\n")

    return results


# ==============================================
# MAIN EXPERIMENT FUNCTION
# ==============================================

def run_single_object_experiment(args):
    """Run single object picking experiment with optional complexity verification."""

    print("\n" + "="*80)
    print("SINGLE OBJECT PICKING EXPERIMENT")
    print("="*80)
    controller_name = args.controller_name or (args.controller.rpartition(":")[2] if args.controller else args.policy)
    print(f"Target Object:    {args.target_object}")
    print(f"Controller:       {controller_name}")
    if args.use_transformer:
        print(f"  Checkpoint:     {args.transformer_checkpoint}")
    elif not args.controller:
        print(f"  Policies dir:   {args.trained_policies_dir}")
    if args.target_complexity:
        print(f"Target Complexity: {args.target_complexity}")
        config = CLUTTER_CONFIGS[args.target_complexity]
        print(f"Description:      {config['description']}")
    else:
        print(f"Clutter Objects:  {args.num_clutter} (random placement)")
    print("="*80 + "\n")

    # Experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = RESULTS_DIR / "benchmark" / controller_name
    if args.target_complexity:
        exp_folder = exp_root / f"single_{args.target_object}_{args.target_complexity}_{timestamp}"
    else:
        exp_folder = exp_root / f"single_{args.target_object}_{timestamp}"

    exp_folder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Experiment folder: {exp_folder.absolute()}")

    exp_classifier_dir = exp_folder / "classifier_images"
    exp_video_dir      = exp_folder / "videos"
    exp_classifier_dir.mkdir(exist_ok=True)
    exp_video_dir.mkdir(exist_ok=True)

    # ── Build environment config ───────────────────────────────────────────────
    env_cfg = BenchmarkEnvCfg()
    env_cfg.scene.num_envs      = 1
    env_cfg.num_trials          = args.num_trials
    env_cfg.max_steps_per_trial = args.max_steps_per_trial
    env_cfg.target_object_id    = args.target_object
    env_cfg.trained_policies_dir = args.trained_policies_dir
    env_cfg.max_drop_attempts   = args.max_attempts
    env_cfg.enable_timeout      = False
    env_cfg.external_controller = bool(args.controller) or args.use_transformer

    # Complexity / clutter config
    if args.target_complexity:
        env_cfg.use_clutter_based_spawn  = True
        env_cfg.target_complexity        = args.target_complexity
        env_cfg.verify_target_complexity = True
        env_cfg.enable_classifier_mode   = True
        env_cfg.classifier_image_dir     = str(exp_classifier_dir)

        config = CLUTTER_CONFIGS[args.target_complexity]
        min_neighbors, max_neighbors = config['num_neighbors']

        if env_cfg.spawn_additional_far_objects:
            min_total = 1 + min_neighbors + env_cfg.num_additional_far_objects
            max_total = 1 + max_neighbors + env_cfg.num_additional_far_objects
        else:
            min_total = 1 + min_neighbors
            max_total = 1 + max_neighbors

        env_cfg.min_objects_to_spawn = min_total
        env_cfg.max_objects_to_spawn = max_total
        print(f"[INFO] Complexity mode: {min_total}-{max_total} total objects")
    else:
        env_cfg.use_clutter_based_spawn  = False
        env_cfg.min_objects_to_spawn     = args.num_clutter + 1
        env_cfg.max_objects_to_spawn     = args.num_clutter + 1
        print(f"[INFO] Random mode: {args.num_clutter + 1} total objects")

    if args.isolated:
        print(f"[INFO] ISOLATED MODE enabled - single object only")
        env_cfg.use_isolated_mode        = True
        env_cfg.randomize_object_orientation = True
        env_cfg.min_objects_to_spawn     = 1
        env_cfg.max_objects_to_spawn     = 1

    # ── Create environment ─────────────────────────────────────────────────────
    env = gym.make("ClutterGrasp-Contactile-Benchmark-v0", cfg=env_cfg,
                   render_mode="rgb_array" if args.video else None)

    if args.video:
        env = gym.wrappers.RecordVideo(env,
            video_folder=str(exp_video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    env          = Sb3VecEnvWrapper(env, fast_variant=True)
    unwrapped_env = env.unwrapped.unwrapped

    # ── Complexity verification (classifier mode) ──────────────────────────────
    verified_base_image  = None
    actual_complexity    = None
    verification_success = False

    if args.target_complexity:
        print("\n[EXPERIMENT] Loading classifier for verification...")
        device_cls = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            classifier_model = load_classifier_model(args.classifier_model_dir, device_cls)
            unwrapped_env.set_classifier(classifier_model, predict_complexity_from_image)
            print(f"[SETUP] ✓ Classifier enabled for complexity verification")

            print("\n[EXPERIMENT] Starting complexity verification...")
            verification_success, verified_base_image = verify_target_object_complexity(
                unwrapped_env,
                env_cfg.target_complexity,
                classifier_model,
                args,
                max_attempts=env_cfg.max_spawn_verification_attempts,
            )

            if verification_success:
                print("[EXPERIMENT] ✓ Target complexity verified!")
                actual_complexity = args.target_complexity
            else:
                print("[EXPERIMENT] ⚠ Proceeding without perfect verification")
                actual_complexity = "unverified"

        except Exception as e:
            print(f"[ERROR] Complexity verification failed: {e}")
            import traceback
            traceback.print_exc()
            actual_complexity = "failed"

    # ── Controller setup ───────────────────────────────────────────────────────
    # Any controller only needs act(obs) -> actions; obs/actions are the env's
    # (num_envs, 234) observation and (num_envs, 12) normalised hand action.
    if args.controller:
        controller = load_external_controller(args.controller, unwrapped_env)
        print(f"\n[CONTROLLER] Using external controller: {args.controller}")
    elif args.use_transformer:
        controller = TransformerController(
            unwrapped_env, args.transformer_checkpoint, args.transformer_obs_norm_stats,
            d_model=args.transformer_d_model, nhead=args.transformer_nhead,
            num_layers=args.transformer_num_layers, dim_feedforward=args.transformer_dim_feedforward,
            num_proprio_tokens=args.transformer_num_proprio_tokens,
        )
    else:
        # Per-object PPO: the env loaded <trained_policies_dir>/<object>/model.zip.
        policy = unwrapped_env.get_current_policy()
        if policy is None:
            print("[ERROR] Failed to load policy!")
            env.close()
            return

        vecnorm_path = unwrapped_env.get_current_vecnorm_path()
        if vecnorm_path and Path(vecnorm_path).exists():
            try:
                # ── Validate obs dim before wrapping ──────────────────────────
                with open(vecnorm_path, "rb") as _f:
                    _vn_data = pickle.load(_f)
                pkl_obs_dim = None
                if hasattr(_vn_data, "obs_rms"):
                    pkl_obs_dim = int(_vn_data.obs_rms.mean.shape[0])
                elif isinstance(_vn_data, dict) and "obs_rms" in _vn_data:
                    pkl_obs_dim = int(_vn_data["obs_rms"].mean.shape[0])

                env_obs_dim = unwrapped_env.cfg.observation_space

                if pkl_obs_dim is not None and pkl_obs_dim != env_obs_dim:
                    print(f"[WARN] VecNormalize obs dim MISMATCH — skipping normalisation!")
                    print(f"[WARN]   pkl obs_dim : {pkl_obs_dim}")
                    print(f"[WARN]   env obs_dim : {env_obs_dim}")
                    print(f"[WARN]   pkl path    : {vecnorm_path}")
                else:
                    base_env = env
                    while hasattr(base_env, 'venv'):
                        base_env = base_env.venv

                    vecnorm             = VecNormalize.load(vecnorm_path, base_env)
                    vecnorm.training    = False
                    vecnorm.norm_reward = False

                    env = vecnorm
                    print(f"[INFO] ✓ VecNormalize loaded  (obs_dim={env_obs_dim})")
                    print(f"[INFO]   pkl: {vecnorm_path}")
            except Exception as e:
                print(f"[WARN] Could not load VecNormalize: {e}")

        controller = PPOController(policy)
        print(f"\n[CONTROLLER] Using PPO policy: {unwrapped_env._object_infos[0].object_id}")

    obs = env.reset()

    # ── Main loop ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"STARTING PICKING EXPERIMENT — {env_cfg.num_trials} TRIALS")
    print(f"{'='*80}\n")

    start_time  = time.time()
    total_steps = 0
    max_steps   = args.video_length

    while total_steps < max_steps:
        actions = controller.act(obs)
        obs, rewards, dones, infos = env.step(actions)
        total_steps += 1

        if dones.any():
            break

        if total_steps % 1000 == 0:
            current_trial = unwrapped_env._current_trial + 1
            trial_steps   = unwrapped_env._trial_step_counter
            print(f"[PROGRESS] Trial {current_trial}/{env_cfg.num_trials}, "
                  f"Steps: {trial_steps}, Total: {total_steps}")

    end_time   = time.time()
    total_time = end_time - start_time

    # ── Collect results ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"CALCULATING RESULTS")
    print(f"{'='*80}\n")

    trial_results = unwrapped_env._trial_results

    print(f"Calculating chaos metrics for {len(trial_results)} trials...")

    for trial_idx, trial_result in enumerate(trial_results):
        trial_num = trial_idx + 1

        if (trial_idx < len(unwrapped_env._trial_initial_positions) and
                trial_idx < len(unwrapped_env._trial_final_positions)):
            initial_pos = unwrapped_env._trial_initial_positions[trial_idx]
            final_pos   = unwrapped_env._trial_final_positions[trial_idx]

            chaos_metrics = calculate_scene_chaos(
                initial_positions=initial_pos,
                final_positions=final_pos,
                object_status={0: 'picked' if trial_result['success'] else 'failed'}
            )
            trial_result['chaos_metrics'] = chaos_metrics
            print(f"  Trial {trial_num}: Chaos = {chaos_metrics['target_distance']:.4f}m "
                  f"({chaos_metrics['target_normalized_distance']:.4f} normalized)")
        else:
            print(f"  Trial {trial_num}: Missing position data!")
            trial_result['chaos_metrics'] = None

    print(f"✓ Chaos calculation complete\n")

    # Overall statistics
    num_trials        = len(trial_results)
    successful_trials = [t for t in trial_results if t['success']]
    failed_trials     = [t for t in trial_results if not t['success']]

    def _mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    overall_stats = {
        'total_trials':     num_trials,
        'successful_trials': len(successful_trials),
        'failed_trials':    len(failed_trials),
        'success_rate':     len(successful_trials) / num_trials if num_trials > 0 else 0.0,

        'avg_drops':        _mean([t['drops']          for t in trial_results]),
        'avg_hand_respawns':_mean([t['hand_respawns']  for t in trial_results]),
        'avg_steps':        _mean([t['steps']          for t in trial_results]),
        'avg_picking_time': _mean([t['picking_time']   for t in trial_results]),

        'avg_drops_success':        _mean([t['drops']         for t in successful_trials]),
        'avg_hand_respawns_success':_mean([t['hand_respawns'] for t in successful_trials]),
        'avg_steps_success':        _mean([t['steps']         for t in successful_trials]),
        'avg_picking_time_success': _mean([t['picking_time']  for t in successful_trials]),

        'avg_chaos_distance':  0.0,
        'avg_chaos_normalized': 0.0,
    }

    chaos_distances  = [t['chaos_metrics']['target_distance']
                        for t in trial_results if t.get('chaos_metrics')]
    chaos_normalized = [t['chaos_metrics']['target_normalized_distance']
                        for t in trial_results if t.get('chaos_metrics')]
    if chaos_distances:
        overall_stats['avg_chaos_distance']  = float(np.mean(chaos_distances))
    if chaos_normalized:
        overall_stats['avg_chaos_normalized'] = float(np.mean(chaos_normalized))

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS")
    print(f"{'='*80}")
    print(f"Target Object:  {args.target_object}")
    print(f"Controller:     {controller_name}")
    if args.target_complexity:
        print(f"Complexity:     {args.target_complexity}  (verified: {actual_complexity})")
    print(f"\nOverall Statistics:")
    print(f"  Total Trials:   {overall_stats['total_trials']}")
    print(f"  Successful:     {overall_stats['successful_trials']}")
    print(f"  Failed:         {overall_stats['failed_trials']}")
    print(f"  Success Rate:   {overall_stats['success_rate']:.1%}")
    print(f"\nAverage Metrics (All Trials):")
    print(f"  Drops:          {overall_stats['avg_drops']:.2f}")
    print(f"  Hand Respawns:  {overall_stats['avg_hand_respawns']:.2f}")
    print(f"  Steps:          {overall_stats['avg_steps']:.1f}")
    print(f"  Picking Time:   {overall_stats['avg_picking_time']:.2f}s")
    print(f"  Chaos Distance: {overall_stats['avg_chaos_distance']:.4f}m")
    print(f"  Chaos Norm:     {overall_stats['avg_chaos_normalized']:.4f}")

    if successful_trials:
        print(f"\nAverage Metrics (Successful Trials Only):")
        print(f"  Drops:          {overall_stats['avg_drops_success']:.2f}")
        print(f"  Hand Respawns:  {overall_stats['avg_hand_respawns_success']:.2f}")
        print(f"  Steps:          {overall_stats['avg_steps_success']:.1f}")
        print(f"  Picking Time:   {overall_stats['avg_picking_time_success']:.2f}s")

    print(f"\nTotal Experiment Time: {total_time:.2f}s")
    print(f"Total Steps:           {total_steps}")

    print(f"\n{'='*80}")
    print(f"PER-TRIAL BREAKDOWN")
    print(f"{'='*80}")
    for trial_result in trial_results:
        status = "✓ SUCCESS" if trial_result['success'] \
            else f"✗ FAILED ({trial_result.get('reason', 'unknown')})"
        print(f"\nTrial {trial_result['trial']}: {status}")
        print(f"  Drops:         {trial_result['drops']}")
        print(f"  Hand Respawns: {trial_result['hand_respawns']}")
        print(f"  Steps:         {trial_result['steps']}")
        print(f"  Time:          {trial_result['picking_time']:.2f}s")
        if trial_result.get('chaos_metrics'):
            cm = trial_result['chaos_metrics']
            ip = cm.get('initial_pos', {}).get(0, (0, 0))
            fp = cm.get('final_pos',   {}).get(0, (0, 0))
            print(f"  Chaos:         {cm['target_distance']:.4f}m "
                  f"({cm['target_normalized_distance']:.4f} normalized)")
            print(f"  Initial pos:   ({ip[0]:.4f}, {ip[1]:.4f})")
            print(f"  Final pos:     ({fp[0]:.4f}, {fp[1]:.4f})")
    print(f"{'='*80}\n")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "timestamp":            datetime.now().isoformat(),
        "target_object":        args.target_object,
        "controller":           controller_name,
        "target_complexity":    args.target_complexity,
        "actual_complexity":    actual_complexity if args.target_complexity else None,
        "num_clutter":          len(unwrapped_env.objects) - 1,
        "num_trials":           env_cfg.num_trials,
        "trial_results":        trial_results,
        "overall_statistics":   overall_stats,
        "total_experiment_time": total_time,
        "total_steps":          total_steps,
        "max_attempts_per_trial": args.max_attempts,
    }

    config_dict = {
        "timestamp":            datetime.now().isoformat(),
        "target_object":        args.target_object,
        "controller":           controller_name,
        "controller_spec":      args.controller or None,
        "transformer_checkpoint": args.transformer_checkpoint if args.use_transformer else None,
        "target_complexity":    args.target_complexity,
        "num_clutter":          args.num_clutter if not args.target_complexity else len(unwrapped_env.objects) - 1,
        "trained_policies_dir": None if (args.controller or args.use_transformer) else args.trained_policies_dir,
        "classifier_model_dir": args.classifier_model_dir if args.target_complexity else None,
        "max_attempts":         args.max_attempts,
        "video_enabled":        args.video,
        "complexity_verified":  verification_success if args.target_complexity else None,
    }

    config_path = exp_folder / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    save_simple_results(exp_folder, results)

    print(f"\n[INFO] ✓ All data saved to: {exp_folder.absolute()}")
    print(f"[INFO]   - config.json")
    print(f"[INFO]   - results/results.json")
    print(f"[INFO]   - results/trial_results.json")
    if args.target_complexity:
        print(f"[INFO]   - classifier_images/")
    if args.video:
        print(f"[INFO]   - videos/")

    env.close()


# ==============================================
# ENTRY POINT
# ==============================================

if __name__ == "__main__":
    try:
        run_single_object_experiment(args_cli)
    except KeyboardInterrupt:
        print("\n[INFO] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()