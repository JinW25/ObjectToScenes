# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Panda + parallel-gripper baseline of the clutter grasping protocol for one object.

No learned controller: the arm is driven by a scripted IK state machine inside the env, and
the grasp pose comes from GG-CNN (optionally with SAM to localise the target). It needs no
trained policy, which makes it the quickest way to check that the benchmark runs.

Usage:
    /isaac-sim/python.sh scripts/benchmark/run_panda_ik.py --target_object A24_0 --isolated --enable_cameras
    /isaac-sim/python.sh scripts/benchmark/run_panda_ik.py --target_object A24_0 --target_complexity C1_medium --enable_cameras

Results: results/benchmark/panda_ik/single_<object>[_<condition>]_<timestamp>/results/results.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

# Repository layout (see README.md): this file lives in scripts/benchmark/.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
WEIGHTS_DIR = REPO_ROOT / "weights"

import numpy as np
import torch
import gymnasium as gym

# ── Isaac Lab app launcher boilerplate (same pattern as run_benchmark_experiment.py) ──
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Panda IK pick-and-lift benchmark")
parser.add_argument("--target_object", type=str, required=True, help="Object ID to pick (matches a USD filename)")
parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument("--num_clutter", type=int, default=0, help="Number of passive clutter objects")
parser.add_argument("--target_complexity", type=str, default="", choices=["", "C0_easy", "C1_medium", "C2_hard"])
parser.add_argument("--isolated", action="store_true", help="Single object only, no clutter")
parser.add_argument("--restrict_to_yaw", action="store_true", default=True,
                     help="Only randomize object yaw (default, fair for a top-down parallel gripper)")
parser.add_argument("--full_orientation_random", action="store_true",
                     help="Match ContactileHand env's fully random roll/pitch/yaw (harder, lower success rate)")
parser.add_argument("--object_usd_dir", type=str, default=None)
parser.add_argument("--ggcnn_checkpoint", type=str, default=str(WEIGHTS_DIR / "ggcnn" / "ggcnn_epoch_23_cornell_statedict.pt"),
                     help="Path to a trained GG-CNN checkpoint (torch state_dict matching "
                          "ggcnn_model.GGCNN). Empty (default) disables GG-CNN and falls back "
                          "to naive object-root grasping every trial. Requires --enable_cameras.")
parser.add_argument("--sam_checkpoint", type=str, default=str(WEIGHTS_DIR / "sam" / "sam_vit_b_01ec64.pth"),
                     help="Path to a SAM checkpoint (e.g. sam_vit_b_01ec64.pth). Empty (default) "
                          "disables SAM and falls back to whole-frame GG-CNN candidate matching "
                          "(no crop). Requires --ggcnn_checkpoint AND --enable_cameras to do anything.")
parser.add_argument("--sam_model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                     help="Must match the checkpoint's backbone")
parser.add_argument("--save_scene_images", action="store_true", default=False,
                     help="Save one raw (undecorated) rgb+depth snapshot of the scene from the wrist "
                          "camera, on trial 1 only, to <exp_folder>/scene_images/.")
parser.add_argument("--video", action="store_true", default=False,
                     help="Record video. Captures the main viewport (a third-person overview of the whole "
                          "scene), NOT the wrist camera's own top-down feed used for grasping.")
parser.add_argument("--video_length", type=int, default=100000, help="Video length in steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Enable cameras if video was requested -- matches run_benchmark_experiment.py's own handling.
# Must happen before AppLauncher(args) launches the sim.
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Everything Isaac-Lab-dependent must be imported after AppLauncher starts ──
import clutter_grasp.envs  # noqa: F401  (registers the gym environments)
from clutter_grasp.envs.panda_ik_benchmark_env import PandaIKBenchmarkEnv
from clutter_grasp.envs.panda_ik_benchmark_env_cfg import PandaIKBenchmarkEnvCfg


def save_results(exp_folder: Path, results: dict):
    results_dir = exp_folder / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    trial_list = results["trial_results"]
    with open(results_dir / "trial_results.json", "w") as f:
        json.dump(trial_list, f, indent=2)

    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    successful_trials = [t for t in trial_list if t["success"]]
    successes = len(successful_trials)

    # Same shape as run_benchmark_experiment.py's overall_stats block: overall
    # averages across every trial, plus the same averages restricted to
    # successful trials only.
    overall_stats = {
        "total_trials": len(trial_list),
        "successful_trials": successes,
        "failed_trials": len(trial_list) - successes,
        "success_rate": successes / max(len(trial_list), 1),

        "avg_drops": _mean([t["drops"] for t in trial_list]),
        "avg_picking_time": _mean([t["picking_time"] for t in trial_list]),

        "avg_drops_success": _mean([t["drops"] for t in successful_trials]),
        "avg_picking_time_success": _mean([t["picking_time"] for t in successful_trials]),
    }
    results["overall_stats"] = overall_stats
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}\nTRIAL SUMMARY\n{'='*80}")
    print(f"Successful trials: {successes}/{len(trial_list)}  "
          f"({100.0 * overall_stats['success_rate']:.1f}%)")
    print(f"Average drops:        {overall_stats['avg_drops']:.2f}")
    print(f"Average picking time: {overall_stats['avg_picking_time']:.2f}s")
    if successful_trials:
        print(f"\nSuccessful trials only:")
        print(f"  Average drops:        {overall_stats['avg_drops_success']:.2f}")
        print(f"  Average picking time: {overall_stats['avg_picking_time_success']:.2f}s")
    print(f"{'='*80}\n")
    print(f"[SAVE] Results saved to: {results_dir}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"single_{args.target_object}_{timestamp}"
    if args.target_complexity:
        folder_name = f"single_{args.target_object}_{args.target_complexity}_{timestamp}"
    exp_folder = RESULTS_DIR / "benchmark" / "panda_ik" / folder_name
    exp_folder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Experiment folder: {exp_folder.absolute()}")

    exp_video_dir = exp_folder / "videos"
    exp_video_dir.mkdir(exist_ok=True)

    env_cfg = PandaIKBenchmarkEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.num_trials = args.num_trials
    env_cfg.target_object_id = args.target_object
    env_cfg.randomize_object_yaw_only = not args.full_orientation_random
    if args.object_usd_dir:
        env_cfg.object_usd_dir = args.object_usd_dir
    env_cfg.ggcnn_checkpoint = args.ggcnn_checkpoint
    env_cfg.sam_checkpoint = args.sam_checkpoint
    env_cfg.sam_model_type = args.sam_model_type
    # Debug images (grasp decisions + SAM detections) go INSIDE this run's own
    # experiment folder, not the fixed source-tree-relative default -- so they
    # land next to results.json for this specific run instead of a shared
    # ggcnn_debug_images/ folder that every run dumps into indiscriminately.
    env_cfg.ggcnn_debug_dir = str(exp_folder / "debug_images")
    env_cfg.save_scene_images = args.save_scene_images
    env_cfg.scene_image_dir = str(exp_folder / "scene_images")

    if args.target_complexity:
        env_cfg.use_clutter_based_spawn = True
        env_cfg.target_complexity = args.target_complexity
        print(f"[INFO] Complexity mode: {args.target_complexity}")
    elif args.isolated:
        env_cfg.use_isolated_mode = True
        print(f"[INFO] Isolated mode: target only, no clutter")
    else:
        env_cfg.min_objects_to_spawn = args.num_clutter + 1
        env_cfg.max_objects_to_spawn = args.num_clutter + 1
        print(f"[INFO] Random mode: {args.num_clutter} clutter object(s)")

    print(f"\n{'='*80}\nPANDA + PARALLEL GRIPPER -- PRIVILEGED IK BENCHMARK\n{'='*80}")
    print(f"Target object:  {args.target_object}")
    print(f"Controller:     Differential IK (scripted, no learning)")
    print(f"Grasp pose:     {'GG-CNN (' + args.ggcnn_checkpoint + ')' if args.ggcnn_checkpoint else 'naive (object root)'}")
    print(f"SAM localizer:  {'ENABLED (' + args.sam_checkpoint + ')' if args.sam_checkpoint else 'disabled (whole-frame GG-CNN matching, no crop)'}")
    print(f"Trials:         {args.num_trials}")
    print(f"{'='*80}\n")

    env = PandaIKBenchmarkEnv(cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(exp_video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video (main viewport) to: {exp_video_dir}")

    # env.device only exists on the bare PandaIKBenchmarkEnv -- once RecordVideo
    # wraps it, that attribute isn't proxied through, so this must go through
    # .unwrapped (matches run_benchmark_experiment.py's own unwrapped_env.device
    # usage, minus its extra .unwrapped.unwrapped, which is only needed there
    # because of an additional Sb3VecEnvWrapper layer that this script doesn't have).
    dummy_action = torch.zeros(
        (env_cfg.scene.num_envs, env_cfg.action_space), dtype=torch.float32, device=env.unwrapped.device
    )
    env.reset()

    while env.unwrapped._current_trial <= args.num_trials:
        _, _, terminated, truncated, _ = env.step(dummy_action)
        if terminated.any() or truncated.any():
            # DirectRLEnv auto-resets internally on the next step; the outcome of the
            # trial that just ended is captured inside _reset_idx.
            pass

    trial_results = env.unwrapped.get_all_trial_results()[: args.num_trials]
    results = {
        "target_object": args.target_object,
        "num_trials": args.num_trials,
        "controller": "panda_parallel_gripper_ik",
        "target_complexity": args.target_complexity or None,
        "trial_results": trial_results,
    }
    save_results(exp_folder, results)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()