# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Contactile-hand clutter benchmark (one target object, C0/C1/C2 clutter or isolated)."""

from __future__ import annotations
import os
import math
import numpy as np
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from clutter_grasp.assets.contactile_hand import CONTACTILE_HAND_CFG

# Object USD directory (same format as training env)
from clutter_grasp.paths import OBJECTS_DIR, WEIGHTS_DIR
USD_DIR = str(OBJECTS_DIR)
POLICY_DIR = str(WEIGHTS_DIR / "ppo_policies")


@dataclass
class ObjectSpawnInfo:
    """Information about an object to spawn."""
    object_id: str
    usd_path: str
    policy_path: str
    vecnorm_path: str

from clutter_grasp.protocol.clutter_levels import CLUTTER_CONFIGS  # noqa: E402  (single definition of the clutter levels)

@configclass
class BenchmarkEnvCfg(DirectRLEnvCfg):
    """Configuration for the Contactile-hand clutter benchmark environment.
    
    This environment spawns multiple objects on a table and sequentially picks them up
    using their corresponding trained policies. Once an object is successfully picked,
    it's removed and the next object becomes the target.
    """
    
    # Simulation settings (match training)
    decimation = 2
    episode_length_s = 4.0
    
    # Action/observation space (match training)
    action_space = 12
    observation_space = 234  # Will be updated based on point cloud size
    state_space = 0
    asymmetric_obs = False
    
    # Simulation configuration (match training)
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
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            bounce_threshold_velocity=0.5,
            enable_stabilization=True,
            solver_type=1,
        ),
    )
    
    # Scene configuration - SINGLE environment for sequential picking
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # Single environment for sequential task
        env_spacing=2.0,
        replicate_physics=False,
    )
    
    # Table parameters (match training)
    table_width: float = 0.85
    table_depth: float = 0.85
    table_height: float = 0.8
    table_thickness: float = 0.05
    leg_radius: float = 0.03
    
    # Robot configuration (match training)
    robot_cfg: ArticulationCfg = CONTACTILE_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    
    # ============ CAMERA CONFIGURATION ============
    # Camera positioned at 60° angle looking down at table from front
    # Note: Requires --enable_cameras flag to be used
    classifier_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/ClassifierCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -1.6, 1.8),  # 0.5m in front (negative Y), 1.3m above origin
            rot=(0.86603, 0.5, 0.0, 0.0),  # 60° angle looking down (w, x, y, z)
            convention="opengl",
        ),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=640,
        height=480,
    )
    
    # Classifier mode settings
    enable_classifier_mode: bool = False  # Enable classifier-based object selection
    classifier_image_dir: str = "classifier_images"  # Directory to save captured images
    classifier_image_prefix: str = "scene"  # Prefix for saved images
    save_annotated_images: bool = False  # Save images with object annotations
    
    # Actuated joint names (match training)
    actuated_joint_names = [
        "thumb_joint_1",
        "thumb_joint_2",
        "index_joint_1",
        "middle_joint_1",
        "ring_joint_1",
        "pinky_joint_1",
    ]
    
    # Site configuration (match training)
    finger_site_parent_bodies = [
        "thumb_link_2", "thumb_link_2",
        "index_link_1", "index_link_2",
        "middle_link_1", "middle_link_2",
        "ring_link_1", "ring_link_2",
        "pinky_link_1", "pinky_link_2",
    ]
    
    palm_site_parent_bodies = ["root"] * 7
    
    finger_site_usd_paths = [
        "thumb_link_2/sites/thumb_site_1/thumb_site_1",
        "thumb_link_2/sites/thumb_site_2/thumb_site_2",
        "index_link_1/sites/index_site_1/index_site_1",
        "index_link_2/sites/index_site_2/index_site_2",
        "middle_link_1/sites/middle_site_1/middle_site_1",
        "middle_link_2/sites/middle_site_2/middle_site_2",
        "ring_link_1/sites/ring_site_1/ring_site_1",
        "ring_link_2/sites/ring_site_2/ring_site_2",
        "pinky_link_1/sites/pinky_site_1/pinky_site_1",
        "pinky_link_2/sites/pinky_site_2/pinky_site_2",
    ]
    
    palm_site_usd_paths = [
        "root/sites/palm_site_1/palm_site_1",
        "root/sites/palm_site_2/palm_site_2",
        "root/sites/palm_site_3/palm_site_3",
        "root/sites/palm_site_4/palm_site_4",
        "root/sites/palm_site_5/palm_site_5",
        "root/sites/palm_site_6/palm_site_6",
        "root/sites/palm_site_7/palm_site_7",
    ]
    
    # RL Action parameters (match training)
    max_pos_delta: float = 0.8
    max_rot_delta: float = 0.4
    max_finger_delta: float = 0.05
    hand_velocity_smoothing: float = 0.7
    
    # Rotation/position limits (match training)
    rot_limit_rad: float = math.pi / 2
    hand_x_limit: float = 0.5
    hand_y_limit: float = 0.5
    hand_z_min: float = 0.6
    hand_z_max: float = 2.5
    hand_init_height_above_object: float = 0.3
    
    # Point cloud sampling (match training)
    num_object_pc_points: int = 32

    # When True, objects are registered from the USD directory and no per-object
    # PPO policy is loaded: the runner supplies actions from its own controller.
    external_controller: bool = False
    
    # Contact sensor settings (match training)
    enable_contact_sensors: bool = False
    contact_force_threshold: float = 0.1
    max_contact_force: float = 100.0
    contact_force_range: tuple = (0.0, 100.0)
    
    # Reference states (match training)
    ref_palm_euler: tuple = (np.pi, 0.0, 0.0)
    ref_open_joints: tuple = (-1.57, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Tolerances (match training)
    palm_orientation_tolerance: float = np.pi / 6
    open_palm_tolerance: float = 0.1
    grasp_distance_threshold: float = 0.06
    
    # Lifting thresholds (match training)
    min_lift_height: float = 0.05
    max_lift_height: float = 0.20
    object_fall_margin: float = 0.05
    
    # Distance thresholds (match training)
    thumb_threshold: float = 0.015
    fingers_threshold: float = 0.02
    palm_threshold: float = 0.03
    
    # Reward weights (match training)
    alpha_hands: float = -2.0
    alpha_sensors: float = 0.5
    alpha_force: float = 2.0
    alpha_open: float = -0.01
    alpha_rot: float = -0.05
    alpha_grasp: float = 5.0
    alpha_thumb_close: float = -1.0
    alpha_fingers_close: float = -1.0
    alpha_palm_close: float = -1.0
    alpha_lift: float = 50.0
    alpha_success: float = 100.0
    alpha_wrap: float = -1.0
    
    # Success criteria (match training)
    success_tolerance: float = 0.01
    max_consecutive_success: int = 50
    
    # Debug settings
    enable_debug_obs: bool = False
    enable_debug_reward: bool = False
    debug_print_interval: int = 100
    visualize_point_cloud: bool = False
    visualize_env_count: int = 1
    verbose_pointcloud_extraction: bool = False

    # Verify spawn selection
    verify_spawn_selection: bool = False

    # Retry behavior
    allow_retry_on_table_fall: bool = True  # Allow retry if object falls on table (not off)
    immediate_hand_reset_on_fail: bool = True  # Immediately reset hand when object falls off table
    
    # Noise models (match training)
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="abs"),
    )
    
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.00005, operation="abs"),
    )
    
    # ============ Multi-Object Sequential Picking Parameters ============

    # ============ SINGLE TARGET OBJECT MODE ============
    target_object_id: str = ""  # Specific object ID to pick
    target_object_must_spawn: bool = True  # Ensure target object is in the scene
    single_object_mode: bool = False  # Enable single-target mode (vs sequential clearing)
    # ==================================================
    
    # ============ ISOLATED MODE ============
    use_isolated_mode: bool = False  # Enable single object mode (no clutter)
    isolated_spawn_area_x: tuple = (-0.3, 0.3)  # X range for object spawn
    isolated_spawn_area_y: tuple = (-0.3, 0.3)  # Y range for object spawn
    # =======================================

    # ============ CLUTTER-BASED SPAWNING ============
    use_clutter_based_spawn: bool = False  # Enable complexity-based clutter
    target_complexity: str = "C1_medium"   # Desired complexity: C0_easy, C1_medium, C2_hard
    max_spawn_verification_attempts: int = 20  # Max attempts to achieve desired complexity
    verify_target_complexity: bool = True  # Verify with classifier before starting
    spawn_additional_far_objects: bool = False  # Add distant objects beyond neighbors
    num_additional_far_objects: int = 5   # Number of far objects (total scene realism)
    # ==================================================

    # Number of objects to spawn (range)
    min_objects_to_spawn: int = 18
    max_objects_to_spawn: int = 18

    # Hand repositioning
    smooth_hand_reset: bool = True  # Smoothly move hand instead of teleporting
    hand_reset_speed: float = 0.5  # Speed for smooth repositioning (m/s)
    hand_reset_timeout_steps: int = 200  # Max steps for repositioning before timeout

    # Object spawn area and spacing
    spawn_area_margin: float = 0.15  # Margin from table edge (meters)
    min_object_spacing: float = 0.001  # Minimum distance between objects (meters)
    max_object_spacing: float = 0.01  # Maximum distance between objects (meters)

    # ============ PREDEFINED SCENE CONFIGURATIONS ============
    use_predefined_scene: bool = False  # Use predefined scene layout instead of random
    predefined_scene_name: str = "scene1"  # Which predefined scene to use
    # Available scenes: "scene1", "scene2", "scene3", "random"
    
    # Trial configuration
    num_trials: int = 10  # Number of trials per experiment
    max_steps_per_trial: int = 200  # Max steps before respawning hand
    detect_hand_flip: bool = True  # Detect upside-down hand
    hand_flip_threshold: float = 3*np.pi/4  # 90 degrees from reference (flip detection)
    respawn_hand_on_flip: bool = True  # Auto-respawn if hand flips without lifting object

    # Deterministic scene settings
    scene_random_seed: int = 42  # Random seed for deterministic scenes
    
    # Spawn patterns (used when use_predefined_scene=False)
    randomize_spawn_positions: bool = True  # Random vs grid placement
    randomize_spawn_order: bool = True  # Random object selection (greedy mode when False)
    allow_object_overlap: bool = True  # Allow closer spacing (uses min_spacing only)

    # Spawn validation
    max_spawn_attempts: int = 10  # Maximum attempts to get valid spawn configuration
    spawn_settling_steps: int = 100  # Physics steps to let objects settle after spawn
    spawn_height_tolerance: float = 0.01  # Max deviation from table height (meters)
    max_spawn_distance_from_origin: float = 0.35  # Max distance from table center for valid spawn (meters)

    # Asset directories
    trained_policies_dir: str = POLICY_DIR
    """Per-object PPO policies: <trained_policies_dir>/<object_id>/model.zip (+ model_vecnormalize.pkl)."""
    object_usd_dir: str = USD_DIR # Directory containing object USD files
    
    # Available objects (will be populated at runtime)
    available_objects: list[ObjectSpawnInfo] = field(default_factory=list)
    
    # Time to wait after successful pick before removing object (steps)
    removal_delay_steps: int = 50
    
    # Time to wait after removal before spawning next object (steps)
    spawn_delay_steps: int = 30
    
    # Maximum attempts per object before skipping
    max_attempts_per_object: int = 3
    max_trial_timesteps: int = 1000  # Hard cap: fail trial if this many steps reached
    
    # Randomization settings
    randomize_object_orientation: bool = True
    randomize_spawn_order: bool = True