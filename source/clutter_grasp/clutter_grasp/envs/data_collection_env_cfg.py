# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for lightweight data collection environment — objects + camera only, no robot."""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from clutter_grasp.paths import DATA_DIR

# EGAD objects converted to USD (not shipped: download the EGAD meshes from
# https://dougsm.github.io/egad/ and convert them to USD, see classifier/README.md).
# Override with DataCollectionEnvCfg.object_usd_dir or --usd_dir in the collector script.
USD_DIR = str(DATA_DIR / "egad_usd")


@dataclass
class ObjectSpawnInfo:
    object_id: str
    usd_path:  str


@configclass
class DataCollectionEnvCfg(DirectRLEnvCfg):
    """Minimal env: table + objects + camera. No robot, no policies."""

    # Dummy RL dims (required by DirectRLEnv base class)
    decimation        = 2
    episode_length_s  = 4.0
    action_space      = 1
    observation_space = 1
    state_space       = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,
            bounce_threshold_velocity=0.5,
            enable_stabilization=True,
            solver_type=1,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=False,
    )

    # Table geometry (match sequential env)
    table_width:     float = 0.85
    table_depth:     float = 0.85
    table_height:    float = 0.8
    table_thickness: float = 0.05
    leg_radius:      float = 0.03

    # Camera (same position/intrinsics as sequential env classifier camera)
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/DataCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -1.6, 1.8),
            rot=(0.86603, 0.5, 0.0, 0.0),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=640,
        height=480,
    )

    # Object spawning
    num_object_pc_points:    int   = 32
    min_objects_to_spawn:    int   = 5
    max_objects_to_spawn:    int   = 30
    spawn_settling_steps:    int   = 150
    spawn_area_margin:       float = 0.15
    min_object_spacing:      float = 0.02
    max_spawn_attempts:      int   = 5
    max_spawn_distance_from_origin: float = 0.35
    spawn_height_tolerance:  float = 0.02
    randomize_object_orientation: bool = True

    # USD asset directory (one <EGAD id>.usd per object, e.g. A00_0.usd)
    object_usd_dir: str = USD_DIR

    # Available objects populated at runtime
    available_objects: list[ObjectSpawnInfo] = field(default_factory=list)

    # ── Domain randomisation ──────────────────────────────────────────────────
    # Camera jitter is currently disabled — projection must match capture exactly.
    # Photometric augmentation is applied at the image level by the collector script
    # (classifier/isaac/collect_classifier_dataset.py) instead.
    camera_pos_jitter:   float = 0.0   # reserved for future use
    camera_angle_jitter: float = 0.0   # reserved for future use