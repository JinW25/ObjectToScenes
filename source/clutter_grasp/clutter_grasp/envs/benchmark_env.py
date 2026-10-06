# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
COORDINATE SYSTEM CONVENTION (WORLD-BASED):
============================================
ALL positions, comparisons, and storage use WORLD coordinates consistently.

Key principles:
1. Storage: object_init_pos, current_pos stored in WORLD coordinates
2. Observations: Provide RELATIVE coordinates (world - env_origin) for policy
3. Rewards: Use WORLD coordinates for height comparisons
4. Termination: Use WORLD coordinates with per-env table heights
5. Point clouds: Computed in WORLD coordinates
6. Site positions: Computed in WORLD coordinates

This ensures:
- No double-offset bugs
- Consistent physics behavior across all environments
- Clear separation between storage (world) and policy input (relative)
"""

from __future__ import annotations

import math
import numpy as np
import torch
from collections.abc import Sequence
import os
import cv2
from datetime import datetime
from pathlib import Path
from PIL import Image
import time

import omni.usd
import omni.physx
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from scipy.spatial import ConvexHull

from pathlib import Path
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from .benchmark_env_cfg import BenchmarkEnvCfg, ObjectSpawnInfo
from clutter_grasp.protocol.classifier import estimate_3d_position_from_bbox
from clutter_grasp.protocol.clutter_levels import apply_complexity_correction as _apply_complexity_correction


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Safely normalize vectors with NaN protection."""
    # Check for NaN/Inf first
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"[ERROR] NaN/Inf detected before normalization!")
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    norm = torch.norm(x, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)  # Prevent division by zero
    normalized = x / norm
    
    # Verify result
    if torch.isnan(normalized).any():
        print(f"[ERROR] NaN produced by normalization!")
        # Return unit vector as fallback
        result = torch.zeros_like(x)
        if dim == -1:
            result[..., 0] = 1.0
        return result
    
    return normalized


def check_tensor_validity(tensor: torch.Tensor, name: str, replace_invalid: bool = True) -> torch.Tensor:
    """Check tensor for NaN/Inf and optionally replace with safe values."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        print(f"[WARN] Invalid values detected in {name}:")
        if has_nan:
            print(f"  - NaN count: {torch.isnan(tensor).sum().item()}")
        if has_inf:
            print(f"  - Inf count: {torch.isinf(tensor).sum().item()}")
        
        if replace_invalid:
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            print(f"  - Replaced invalid values in {name}")
    
    return tensor

def extract_policy_pointcloud_for_object(env, obj_idx: int, obj_info):
    """Extract point cloud for object."""
    import torch
    from .benchmark_env import quat_apply_batch
    
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
        
        # DIRECT CALL - no state swapping needed
        mesh_points_local = env._extract_object_mesh_for_current(obj_idx)
        
        if mesh_points_local is None:
            return None
        
        if not isinstance(mesh_points_local, torch.Tensor):
            mesh_points_local = torch.tensor(mesh_points_local, device=env.device, dtype=torch.float32)
        
        current_object = env.objects[obj_idx]
        current_object.update(dt=env.cfg.sim.dt)
        
        obj_pos_world = current_object.data.root_pos_w[0].clone()
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
    import torch
    import numpy as np
    
    cam_pos_w = cam_pos_w.to(dtype=torch.float32)
    cam_quat_w = cam_quat_w.to(dtype=torch.float32)
    
    rot_mat = quat_to_rotation_matrix_ros(cam_quat_w)
    points_rel = points_world - cam_pos_w
    points_cam = torch.matmul(points_rel, rot_mat.T.T)
    
    valid_mask = points_cam[:, 2] > 0.01
    
    if valid_mask.sum() == 0:
        return None
    
    points_cam_valid = points_cam[valid_mask]
    
    width = camera_cfg.width
    height = camera_cfg.height
    
    focal_length_mm = camera_cfg.spawn.focal_length
    h_aperture_mm = camera_cfg.spawn.horizontal_aperture
    
    f_x = (focal_length_mm / h_aperture_mm) * width
    f_y = f_x
    
    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0
    
    xs = f_x * (points_cam_valid[:, 0] / points_cam_valid[:, 2]) + c_x
    ys = f_y * (points_cam_valid[:, 1] / points_cam_valid[:, 2]) + c_y
    
    xs = xs.cpu().numpy()
    ys = ys.cpu().numpy()
    
    return xs, ys

def quat_to_rotation_matrix_ros(quat):
    """Convert ROS quaternion to rotation matrix."""
    import torch
    
    quat = quat.to(dtype=torch.float32)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], device=quat.device, dtype=torch.float32)
    
    return R

def calculate_scene_chaos(initial_positions, final_positions, object_status, table_bounds=None):
    """Calculate average normalized movement of objects in the scene.
    
    Modified for single target mode - only tracks target object movement.
    Handles None values gracefully.
    """
    # Use constant table dimensions
    if table_bounds is None:
        table_size = 0.85  # 0.85m x 0.85m table
        x_min, x_max = -table_size/2, table_size/2
        y_min, y_max = -table_size/2, table_size/2
        table_diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    else:
        x_min, x_max, y_min, y_max = table_bounds
        table_diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    
    # Handle None values
    if initial_positions is None or final_positions is None:
        return {
            'target_distance': 0.0,
            'target_normalized_distance': 0.0,
            'target_dx': 0.0,
            'target_dy': 0.0,
            'target_status': 'no_data',
            'table_diagonal': float(table_diagonal),
            'initial_pos': None,
            'final_pos': None,
        }
    
    if 0 not in initial_positions or 0 not in final_positions:
        return {
            'target_distance': 0.0,
            'target_normalized_distance': 0.0,
            'target_dx': 0.0,
            'target_dy': 0.0,
            'target_status': 'no_data',
            'table_diagonal': float(table_diagonal),
            'initial_pos': None,
            'final_pos': None,
        }
    
    # Calculate movement for target object (index 0)
    status = object_status.get(0, 'unknown')
    
    initial_pos = initial_positions[0]
    final_pos = final_positions[0]
    
    # Calculate Euclidean distance
    dx = final_pos[0] - initial_pos[0]
    dy = final_pos[1] - initial_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Normalize by CONSTANT table diagonal
    normalized_distance = distance / table_diagonal if table_diagonal > 0 else 0.0
    
    chaos_metrics = {
        'target_distance': float(distance),
        'target_normalized_distance': float(normalized_distance),
        'target_dx': float(dx),
        'target_dy': float(dy),
        'target_status': status,
        'table_diagonal': float(table_diagonal),
        'initial_pos': initial_positions.copy(),  # Store full dict
        'final_pos': final_positions.copy(),      # Store full dict
    }
    
    return chaos_metrics

class BenchmarkEnv(DirectRLEnv):
    """Environment for Contactile Hand manipulation with RL-style normalized actions.
    
    Uses WORLD coordinate system consistently throughout.
    """

    cfg: BenchmarkEnvCfg

    def __init__(self, cfg: BenchmarkEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg

        # Simplified state machine
        self._state = "PICKING"
        self._current_object_drops = 0
        self._removal_timer = 0
        
        # Initialize self.objects BEFORE super().__init__()
        self.objects = []
        self._object_infos = []
        self._current_object_idx = 0
        self._picking_order = [0]  # Single target mode: always pick object 0
        self._picked_objects = set()
        self._object_final_status = {}
        self._object_drop_counts = {}
            
        # Load available policies
        self._load_available_policies()
        self._select_objects_to_spawn()

        self._trial_start_time = None  # Wall clock time when trial starts
        self._picking_start_step = 0   # Simulation step when picking starts
        self._classifier_model = None
        
        # Initialize classifier attributes
        self.classifier_camera = None
        self.classifier_image_dir = Path(cfg.classifier_image_dir)
        if cfg.enable_classifier_mode or cfg.use_clutter_based_spawn:
            self.classifier_image_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Classifier image directory: {self.classifier_image_dir.absolute()}")

        self._current_trial = 0
        self._trial_step_counter = 0
        self._trial_results = []  # Store results for each trial
        self._hand_respawn_count = 0  # Count hand respawns within trial

        self._trial_initial_positions = []  # Store initial pos for each trial
        self._trial_final_positions = []    # Store final pos for each trial

        # Call parent init
        super().__init__(cfg, render_mode, **kwargs)
        
        # Retrieve camera from scene
        if self.cfg.enable_classifier_mode or self.cfg.use_clutter_based_spawn:
            if "classifier_camera" in self.scene.sensors:
                self.classifier_camera = self.scene.sensors["classifier_camera"]
                print(f"[INFO] ✓ Classifier camera retrieved from scene")
            else:
                print(f"[WARN] Classifier camera not found in scene")
                self.classifier_camera = None
        
        # Get actuated joint indices
        self._actuated_joint_idx, _ = self.robot.find_joints(self.cfg.actuated_joint_names)
        
        # Initialize site tracking
        self._initialize_site_parent_indices()
        
        # Initialize tracking variables
        self.current_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_euler = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_joint_pos = torch.zeros((self.num_envs, len(self._actuated_joint_idx)), device=self.device)
        self.pos_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.rot_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Success tracking
        self.successes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.object_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.total_episodes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.successful_episodes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # ============ Chaos tracking - positions stored as dicts for compatibility ============
        self.target_initial_pos = None  # Will be dict {0: (x, y)}
        self.target_final_pos = None    # Will be dict {0: (x, y)}
        self.target_status = {}         # Will be dict {0: 'status'}
        # ======================================================================================
        
        self.extras = {} 
        
        # Reference states
        self.ref_palm_euler = torch.tensor(
            self.cfg.ref_palm_euler, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1)
        
        self.ref_open_joints = torch.tensor(
            self.cfg.ref_open_joints, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1)

        # Debug counter
        self._debug_step_counter = 0
        
        # Initialize cluttered scene
        print("[INFO] Setting up cluttered scene with target object...")
        self._initialize_cluttered_scene()
        
        # Initialize sites
        print("[INFO] Performing delayed initialization...")
        self._initialize_site_transforms()

        print(f"[INFO] Initialization complete")
        print(f"[INFO] Target object: {self._object_infos[0].object_id}")
        print(f"[INFO] Clutter objects: {len(self.objects) - 1}")
        
        # Visualization setup if enabled
        if self.cfg.visualize_point_cloud:
            print("\n[INFO] Setting up point cloud visualization...")
            self._pc_markers_initialized = False
            self._site_markers_initialized = False
            self._object_root_markers_initialized = False
            self._num_vis_envs = min(self.cfg.visualize_env_count, self.num_envs)
            
            self._setup_point_cloud_markers()
            self._setup_site_markers()
            self._setup_object_root_markers()
            
            print(f"[INFO] ✓ Point cloud visualization enabled")
        
        # Load policy for target object
        target_obj_info = self._object_infos[0]
        self._load_policy_for_object(target_obj_info)
        
        # Extract mesh for target object
        self._extract_object_mesh_for_current(0)
        
        print(f"[INFO] ✓ Ready to pick target object: {target_obj_info.object_id}")

    def set_classifier(self, classifier_model, predict_complexity_fn):
        """Store classifier model and prediction function for complexity verification.
        
        This must be called after environment initialization if you want to use
        complexity verification during respawning.
        
        Args:
            classifier_model: The trained classifier model
            predict_complexity_fn: Function to predict complexity from image
        """
        self._classifier_model = classifier_model
        self._predict_complexity_fn = predict_complexity_fn
        
        print(f"[CLASSIFIER] ✓ Classifier stored in environment")
        print(f"[CLASSIFIER] Will verify complexity for EVERY trial")

    def _make_site_spheres_rigid_bodies(self):
        """Enable collision and contact reporting on site sphere geometries
        
        The spheres remain as collision shapes but not get proper PhysX contact reporting enabled. 
        ContactSensor will track them by prim path.

        This must be called after robot is spawned but before cloning environments
        """
        stage = omni.usd.get_context().get_stage()

        print("\n" + "="*80)
        print("[SITE SETUP] Configuring site spheres for contact sensing")
        print("="*80)

        # Work on source environment only
        robot_root = "/World/envs/env_0/Robot/root"

        all_paths = self.cfg.finger_site_usd_paths + self.cfg.palm_site_usd_paths
        configured_count = 0

        for site_idx, site_path in enumerate(all_paths):
            full_path = f"{robot_root}/{site_path}"
            prim = stage.GetPrimAtPath(full_path)

            if not prim.IsValid():
                print(f"[WARN] Sphere not found: {full_path}")
                continue

            # Verify that it is a sphere
            if not prim.IsA(UsdGeom.Sphere):
                print(f"[WARN] Not a sphere geometry: {full_path}")
                continue

            # Apply CollisionAPI (making the sphere to participate in collision detection)
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            
            col_api = UsdPhysics.CollisionAPI(prim)
            col_api.CreateCollisionEnabledAttr().Set(True)

            # Apply PhysxContactReportAPI(prim)
            if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim)

            contact_api = PhysxSchema.PhysxContactReportAPI(prim)
            contact_api.CreateThresholdAttr().Set(0.0) # Report all contacts (even light touches)

            # Apply PhysxCollisionAPI (for Physx-specific collision settings)
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)

            physx_col_api = PhysxSchema.PhysxCollisionAPI(prim)
            # Make contact detection more sensitive
            physx_col_api.CreateContactOffsetAttr().Set(0.02)
            physx_col_api.CreateRestOffsetAttr().Set(0.0)

            # Ensure sphere has proper radius
            sphere_geom = UsdGeom.Sphere(prim)
            radius_attr = sphere_geom.GetRadiusAttr()
            if not radius_attr or not radius_attr.Get():
                print(f"[WARN] Sphere missing radius, setting to 0.005m: {full_path}")
                sphere_geom.CreateRadiusAttr().Set(0.005) # 5 mm default
            
            configured_count += 1
        
        print(f"[SITE SETUP] Configured {configured_count}/{len(all_paths)} site spheres")
        print(f"[SITE SETUP] Spheres are collision shapes with contact reporting enabled")
        print("="*80 + "\n")
    
    def _initialize_site_parent_indices(self):
        """Get site parent body indices."""
        unique_finger_parents = list(dict.fromkeys(self.cfg.finger_site_parent_bodies))
        unique_palm_parents = list(dict.fromkeys(self.cfg.palm_site_parent_bodies))
        
        print(f"[INFO] Finding {len(unique_finger_parents)} unique parent bodies for finger sites...")
        unique_finger_idx, unique_finger_names = self.robot.find_bodies(unique_finger_parents)
        
        print(f"[INFO] Finding {len(unique_palm_parents)} unique parent bodies for palm sites...")
        unique_palm_idx, unique_palm_names = self.robot.find_bodies(unique_palm_parents)
        
        finger_parent_map = {name: idx for name, idx in zip(unique_finger_names, unique_finger_idx)}
        palm_parent_map = {name: idx for name, idx in zip(unique_palm_names, unique_palm_idx)}
        
        self._finger_parent_idx = torch.tensor(
            [finger_parent_map[parent] for parent in self.cfg.finger_site_parent_bodies],
            device=self.device,
            dtype=torch.long
        )
        self._palm_parent_idx = torch.tensor(
            [palm_parent_map[parent] for parent in self.cfg.palm_site_parent_bodies],
            device=self.device,
            dtype=torch.long
        )
        
        print(f"[INFO] Mapped {len(self._finger_parent_idx)} finger sites to parent bodies")
        print(f"[INFO] Mapped {len(self._palm_parent_idx)} palm sites to parent bodies")
        
        # Validate parent indices
        max_body_idx = self.robot.num_bodies
        if torch.any(self._finger_parent_idx >= max_body_idx):
            print(f"[ERROR] Invalid finger parent indices! Max allowed: {max_body_idx}")
            self._finger_parent_idx = torch.clamp(self._finger_parent_idx, 0, max_body_idx - 1)
        if torch.any(self._palm_parent_idx >= max_body_idx):
            print(f"[ERROR] Invalid palm parent indices! Max allowed: {max_body_idx}")
            self._palm_parent_idx = torch.clamp(self._palm_parent_idx, 0, max_body_idx - 1)

    def _load_available_policies(self):
        """Register every object that has both a USD file and a trained PPO policy."""
        policies_dir = Path(self.cfg.trained_policies_dir)
        usd_dir      = Path(self.cfg.object_usd_dir)

        print(f"\n[INFO] Scanning for objects...")
        print(f"[INFO] USD directory: {usd_dir.absolute()}")

        # ── General CLUTTER pool ────────────────────────────────────────────────
        # Objects usable as clutter don't need a trained policy -- they just need
        # to physically exist on the table. Built separately from
        # self.cfg.available_objects (restricted to objects that have a trained
        # TARGET policy) so clutter selection isn't limited to the objects that
        # happen to have a policy.
        self._clutter_pool: list = []
        if usd_dir.exists():
            for usd_file in sorted(usd_dir.glob("*.usd")):
                self._clutter_pool.append(ObjectSpawnInfo(
                    object_id   = usd_file.stem,
                    usd_path    = str(usd_file),
                    policy_path = "",
                    vecnorm_path= "",
                ))
        if len(self._clutter_pool) == 0:
            print(f"[WARN] No USD files found in {usd_dir} for clutter pool -- "
                  f"clutter selection will fall back to policy-restricted objects")
        else:
            print(f"[INFO] Clutter pool: {len(self._clutter_pool)} objects available from USD dir")

        # ── External controller: every object in the USD dir, no PPO policies ──
        if self.cfg.external_controller:
            print(f"[INFO] External controller — registering objects from USD dir only")
            if not usd_dir.exists():
                raise FileNotFoundError(f"USD directory not found: {usd_dir}")
            self.cfg.available_objects.extend(self._clutter_pool)
            if len(self.cfg.available_objects) == 0:
                raise RuntimeError(f"No USD files found in: {usd_dir}")
            print(f"[INFO] Registered {len(self.cfg.available_objects)} objects from USD dir")
            return

        # ── Per-object PPO policies: <trained_policies_dir>/<object_id>/model.zip ──
        print(f"[INFO] Scanning for trained policies in: {policies_dir}")
        print(f"[INFO] Policies directory exists: {policies_dir.exists()}")

        if not policies_dir.exists():
            raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

        print(f"[INFO] Contents of policies directory:")
        for item in sorted(policies_dir.iterdir()):
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")

        for policy_dir in sorted(policies_dir.iterdir()):
            if not policy_dir.is_dir():
                continue

            object_id    = policy_dir.name
            policy_path  = policy_dir / "model.zip"
            vecnorm_path = policy_dir / "model_vecnormalize.pkl"

            usd_filename = f"{object_id}.usd"
            usd_path     = usd_dir / usd_filename

            if not policy_path.exists():
                print(f"[WARN] Policy not found for {object_id}, skipping")
                continue

            if not usd_path.exists():
                print(f"[WARN] USD file not found for {object_id}, skipping")
                continue

            obj_info = ObjectSpawnInfo(
                object_id   = object_id,
                usd_path    = str(usd_path),
                policy_path = str(policy_path),
                vecnorm_path= str(vecnorm_path) if vecnorm_path.exists() else "",
            )
            self.cfg.available_objects.append(obj_info)
            print(f"  ✓ Found valid object-policy pair: {object_id}")

        if len(self.cfg.available_objects) == 0:
            print("\n[ERROR] No valid object-policy pairs found!")
            print(f"[ERROR] Searched in policies_dir: {policies_dir.absolute()}")
            print(f"[ERROR] Searched in object_usd_dir: {usd_dir.absolute()}")
            raise RuntimeError("No valid object-policy pairs found!")

        print(f"\n[INFO] Loaded {len(self.cfg.available_objects)} object-policy pairs")

    def _clutter_source_pool(self) -> list:
        """Pool of objects usable as clutter for this scene.

        Prefers self._clutter_pool (built from the full USD directory in
        _load_available_policies -- doesn't require a trained policy). Falls
        back to self.cfg.available_objects (policy-restricted) only if the USD
        scan found nothing, e.g. a misconfigured object_usd_dir.
        """
        pool = getattr(self, "_clutter_pool", None)
        return pool if pool else self.cfg.available_objects

    def _select_objects_to_spawn(self):
        """Select target object and clutter objects.
        
        SUPPORTS:
        - Isolated mode (single object, no clutter)
        - Complexity-based spawning
        - Random spawning
        """
        
        # ============ ISOLATED MODE - SINGLE OBJECT ONLY ============
        if self.cfg.use_isolated_mode:
            print(f"\n{'='*80}")
            print(f"[SPAWN SELECTION] ISOLATED MODE")
            print(f"[SPAWN SELECTION] Single object only - no clutter")
            print(f"{'='*80}")
            
            # Select target
            if self.cfg.target_object_id:
                target_obj_info = None
                for obj_info in self.cfg.available_objects:
                    if obj_info.object_id == self.cfg.target_object_id:
                        target_obj_info = obj_info
                        break
                if target_obj_info is None:
                    raise ValueError(f"Target object '{self.cfg.target_object_id}' not found!")
                print(f"[SPAWN SELECTION] ✓ Specific target: {target_obj_info.object_id}")
            else:
                target_obj_info = np.random.choice(self.cfg.available_objects)
                print(f"[SPAWN SELECTION] ✓ Random target: {target_obj_info.object_id}")
            
            self._object_infos = [target_obj_info]
            
            print(f"{'='*80}\n")
            return  # Exit early - no clutter
        # ============================================================
        
        # ============ CALCULATE NUMBER OF OBJECTS ============
        if self.cfg.use_clutter_based_spawn and self.cfg.target_complexity:
            from .benchmark_env_cfg import CLUTTER_CONFIGS
            
            if self.cfg.target_complexity not in CLUTTER_CONFIGS:
                raise ValueError(f"Invalid target_complexity: {self.cfg.target_complexity}")
            
            config = CLUTTER_CONFIGS[self.cfg.target_complexity]
            min_neighbors, max_neighbors = config['num_neighbors']
            
            # Randomize number of neighbors within range
            num_neighbors = np.random.randint(min_neighbors, max_neighbors + 1)
            
            # Total objects = 1 (target) + neighbors + optional far objects
            if self.cfg.spawn_additional_far_objects:
                num_to_spawn = 1 + num_neighbors + self.cfg.num_additional_far_objects
            else:
                num_to_spawn = 1 + num_neighbors
            
            print(f"\n{'='*80}")
            print(f"[SPAWN SELECTION] COMPLEXITY MODE: {self.cfg.target_complexity}")
            print(f"[SPAWN SELECTION] {config['description']}")
            print(f"[SPAWN SELECTION] Neighbors: {num_neighbors} (range: {min_neighbors}-{max_neighbors})")
            if self.cfg.spawn_additional_far_objects:
                print(f"[SPAWN SELECTION] Far objects: {self.cfg.num_additional_far_objects}")
            print(f"[SPAWN SELECTION] Total objects: {num_to_spawn} (1 target + {num_to_spawn - 1} clutter)")
            print(f"{'='*80}")
        else:
            # Random mode
            num_to_spawn = np.random.randint(
                self.cfg.min_objects_to_spawn,
                self.cfg.max_objects_to_spawn + 1
            )
            
            print(f"\n{'='*80}")
            print(f"[SPAWN SELECTION] RANDOM MODE")
            print(f"[SPAWN SELECTION] Total objects: {num_to_spawn} (1 target + {num_to_spawn - 1} clutter)")
            print(f"{'='*80}")
        
        # ============ TARGET OBJECT SELECTION ============
        target_idx = None
        if self.cfg.target_object_id:
            target_obj_info = None
            for i, obj_info in enumerate(self.cfg.available_objects):
                if obj_info.object_id == self.cfg.target_object_id:
                    target_obj_info = obj_info
                    target_idx = i
                    break
            
            if target_obj_info is None:
                raise ValueError(f"Target object '{self.cfg.target_object_id}' not found!")
            
            print(f"[SPAWN SELECTION] ✓ Target found at available_objects index {target_idx}")
            print(f"[SPAWN SELECTION] Target (index 0): {target_obj_info.object_id}")
        else:
            target_obj_info = np.random.choice(self.cfg.available_objects)
            print(f"[SPAWN SELECTION] Random target (index 0): {target_obj_info.object_id}")
        
        # Start with target at index 0
        self._object_infos = [target_obj_info]
        
        # ============ CLUTTER SELECTION ============
        num_clutter_needed = num_to_spawn - 1
        
        if num_clutter_needed > 0:
            clutter_pool = self._clutter_source_pool()
            # Exclude the target by object_id (not index -- clutter_pool may be
            # a different list than self.cfg.available_objects).
            other_pool = [o for o in clutter_pool if o.object_id != target_obj_info.object_id]
            if len(other_pool) == 0:
                # Only the target itself is available anywhere -- fall back to
                # allowing duplicates of the target as clutter rather than
                # crashing outright.
                print(f"[WARN] No clutter candidates distinct from target "
                      f"'{target_obj_info.object_id}' -- falling back to duplicates")
                other_pool = clutter_pool
            if len(other_pool) == 0:
                raise RuntimeError(
                    f"No objects available for clutter (target='{target_obj_info.object_id}'). "
                    f"Check cfg.object_usd_dir contains USD files."
                )

            if self.cfg.use_predefined_scene or self.cfg.use_clutter_based_spawn:
                # Deterministic clutter selection
                saved_state = np.random.get_state()
                np.random.seed(self.cfg.scene_random_seed)

                clutter_indices = np.random.choice(
                    len(other_pool),
                    size=num_clutter_needed,
                    replace=True
                )

                for i in clutter_indices:
                    self._object_infos.append(other_pool[i])

                np.random.set_state(saved_state)

                if self.cfg.use_clutter_based_spawn:
                    print(f"[SPAWN SELECTION] CLUTTER-BASED mode - added {num_clutter_needed} clutter objects")
                else:
                    print(f"[SPAWN SELECTION] PREDEFINED SCENE mode - added {num_clutter_needed} clutter objects")
            else:
                # Random clutter selection
                clutter_indices = np.random.choice(
                    len(other_pool),
                    size=num_clutter_needed,
                    replace=True
                )

                for i in clutter_indices:
                    self._object_infos.append(other_pool[i])

                print(f"[SPAWN SELECTION] RANDOM mode - added {num_clutter_needed} clutter objects")
        
        # ============ CRITICAL VERIFICATION ============
        if len(self._object_infos) != num_to_spawn:
            raise RuntimeError(
                f"SPAWN BUG: Expected {num_to_spawn} objects but have {len(self._object_infos)}! "
                f"This should never happen."
            )
        
        print(f"\n[SPAWN SELECTION] Final scene composition:")
        print(f"  Index 0 (TARGET): {self._object_infos[0].object_id}")
        for i in range(1, min(len(self._object_infos), 6)):
            print(f"  Index {i} (clutter): {self._object_infos[i].object_id}")
        
        if len(self._object_infos) > 6:
            print(f"  ... and {len(self._object_infos) - 6} more clutter objects")
        
        if self._object_infos[0].object_id != target_obj_info.object_id:
            raise RuntimeError(f"CRITICAL ERROR: Target not at index 0!")
        
        print(f"\n[SPAWN SELECTION] ✓ Verification passed: Target at index 0")
        print(f"[SPAWN SELECTION] ✓ Total scene objects: {len(self._object_infos)}")
        print(f"{'='*80}\n")

    def _setup_classifier_camera(self):
        """Setup camera for classifier mode if enabled.
        
        NOTE: This method only initializes attributes.
        The actual camera is created in _setup_scene() and added to the scene.
        """
        # Always initialize these attributes
        self.classifier_camera = None
        self.classifier_image_dir = Path(self.cfg.classifier_image_dir)
        
        if not self.cfg.enable_classifier_mode:
            return
        
        # Create directory for saving images (camera will be retrieved from scene later)
        self.classifier_image_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Classifier image directory: {self.classifier_image_dir.absolute()}")
        
        # Camera will be added to scene in _setup_scene() and retrieved after super().__init__()
        print("[INFO] Classifier camera will be initialized in scene setup")

    def capture_classifier_image(self) -> str:
        """Capture and save an image from the classifier camera.
        
        Returns:
            str: Path to the saved image file
        """
        if not self.cfg.enable_classifier_mode:
            print("[WARN] Classifier mode not enabled, cannot capture image")
            return ""
        
        if self.classifier_camera is None:
            print("[ERROR] Classifier camera is None - cannot capture")
            return ""
        
        try:
            print("[DEBUG] Step 1: Updating camera...")
            # Update camera (this triggers rendering)
            self.classifier_camera.update(dt=self.cfg.sim.dt)
            print("[DEBUG] ✓ Camera update successful")
            
            print("[DEBUG] Step 2: Getting RGB data...")
            # Check if RGB data exists
            if "rgb" not in self.classifier_camera.data.output:
                print(f"[ERROR] 'rgb' not in camera output")
                print(f"[ERROR] Available outputs: {list(self.classifier_camera.data.output.keys())}")
                return ""
            
            # Get RGB image data
            rgb_data = self.classifier_camera.data.output["rgb"][0]  # First environment
            print(f"[DEBUG] ✓ RGB data retrieved, shape: {rgb_data.shape}, dtype: {rgb_data.dtype}")
            
            print("[DEBUG] Step 3: Converting to numpy...")
            # Convert from tensor to numpy (H, W, C) format
            rgb_np = rgb_data.cpu().numpy()
            print(f"[DEBUG] ✓ Numpy array shape: {rgb_np.shape}, range: [{rgb_np.min():.3f}, {rgb_np.max():.3f}]")
            
            if rgb_np.dtype == np.uint8:
                print(f"[DEBUG] Data already uint8, skipping conversion")
                rgb_uint8 = rgb_np
            else:
                print("[DEBUG] Step 4: Converting float to uint8...")
                # Only convert if it's float [0, 1]
                rgb_uint8 = (rgb_np * 255).astype(np.uint8)
            
            print(f"[DEBUG] ✓ Final array range: [{rgb_uint8.min()}, {rgb_uint8.max()}]")
            
            print("[DEBUG] Step 5: Generating filename...")
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.cfg.classifier_image_prefix}_{timestamp}.png"
            filepath = self.classifier_image_dir / filename
            print(f"[DEBUG] ✓ Will save to: {filepath}")
            
            print("[DEBUG] Step 6: Saving image with PIL (RGB format)...")
            # Save image using PIL to preserve RGB format
            image_pil = Image.fromarray(rgb_uint8, mode='RGB')
            image_pil.save(str(filepath))
            
            print(f"[SUCCESS] ✓✓✓ Image saved successfully in RGB format!")
            print(f"[INFO] Captured classifier image: {filepath}")
            print(f"[INFO] Image size: {rgb_uint8.shape[1]}x{rgb_uint8.shape[0]}")
            return str(filepath)
            
        except KeyError as e:
            print(f"[ERROR] KeyError during capture: {e}")
            print(f"[ERROR] Camera data structure issue")
            import traceback
            traceback.print_exc()
            return ""
        except Exception as e:
            print(f"[ERROR] Failed to capture classifier image: {e}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return ""

    def capture_initial_scene(self):
        """Capture the initial cluttered scene before starting experiment.
        
        This should be called after all objects are spawned and settled.
        """
        if not self.cfg.enable_classifier_mode:
            print("[INFO] Classifier mode not enabled, skipping capture")
            return None
        
        if self.classifier_camera is None:
            print("[ERROR] Cannot capture initial scene - camera is None")
            print("[ERROR] This usually means:")
            print("  1. Camera was not added to scene in _setup_scene()")
            print("  2. --enable_cameras flag not set")
            print("  3. Camera initialization failed")
            
            # Debug: Check if camera is in scene
            if hasattr(self, 'scene') and hasattr(self.scene, 'sensors'):
                print(f"[DEBUG] Available sensors: {list(self.scene.sensors.keys())}")
            
            return None
        
        print("\n" + "="*80)
        print("[CLASSIFIER] Capturing initial scene...")
        print("="*80)
        print(f"[DEBUG] Camera object: {self.classifier_camera}")
        print(f"[DEBUG] Camera type: {type(self.classifier_camera)}")
        
        # Let physics settle and render multiple frames
        print("[INFO] Settling scene before capture (50 steps)...")
        for i in range(50):
            self.sim.step(render=True)  # IMPORTANT: render=True
            
            # Update camera every 10 steps
            if i % 10 == 0:
                try:
                    self.classifier_camera.update(dt=self.cfg.sim.dt)
                    if i == 0:
                        print(f"[DEBUG] Camera update successful (step {i})")
                except Exception as e:
                    print(f"[ERROR] Camera update failed at step {i}: {e}")
        
        # Final camera update before capture
        print("[INFO] Final camera update before capture...")
        try:
            self.classifier_camera.update(dt=self.cfg.sim.dt)
            print("[DEBUG] ✓ Camera update successful")
        except Exception as e:
            print(f"[ERROR] ✗ Camera update failed: {e}")
            print("="*80 + "\n")
            return None
        
        # Check if camera has data
        if not hasattr(self.classifier_camera, 'data'):
            print("[ERROR] Camera has no 'data' attribute")
            print("="*80 + "\n")
            return None
        
        if not hasattr(self.classifier_camera.data, 'output'):
            print("[ERROR] Camera data has no 'output' attribute")
            print("="*80 + "\n")
            return None
        
        print(f"[DEBUG] Camera data outputs: {list(self.classifier_camera.data.output.keys())}")
        
        # Capture image
        print("[INFO] Capturing image now...")
        image_path = self.capture_classifier_image()
        
        if image_path:
            print(f"[SUCCESS] ✓ Initial scene captured: {image_path}")
            print(f"[INFO] Image saved to: {Path(image_path).absolute()}")
            
            # Verify file exists and is not empty
            if Path(image_path).exists():
                file_size = Path(image_path).stat().st_size
                print(f"[INFO] Image file size: {file_size} bytes")
                if file_size == 0:
                    print("[WARN] Image file is empty!")
            else:
                print("[ERROR] Image file does not exist after save!")
            
        else:
            print("[ERROR] ✗ Failed to capture initial scene")
            print("[ERROR] Check console output above for specific error")
        
        print("="*80 + "\n")
        
        return image_path

    def get_latest_classifier_image(self) -> str:
        """Get path to the most recently captured classifier image.
        
        Returns:
            str: Path to latest image, or empty string if none exists
        """
        if not self.cfg.enable_classifier_mode:
            return ""
        
        # Check if directory attribute exists
        if not hasattr(self, 'classifier_image_dir'):
            print("[WARN] Classifier image directory not initialized")
            return ""
        
        try:
            # Check if directory exists
            if not self.classifier_image_dir.exists():
                print(f"[WARN] Classifier image directory does not exist: {self.classifier_image_dir}")
                return ""
            
            image_files = sorted(
                self.classifier_image_dir.glob(f"{self.cfg.classifier_image_prefix}_*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Filter out annotated images
            image_files = [f for f in image_files if not f.name.endswith("_annotated.png")]
            
            if image_files:
                return str(image_files[0])
            else:
                print(f"[WARN] No classifier images found in: {self.classifier_image_dir}")
                return ""
                
        except Exception as e:
            print(f"[ERROR] Failed to get latest image: {e}")
            import traceback
            traceback.print_exc()
            return ""


    def _load_policy_for_object(self, obj_info: ObjectSpawnInfo):
        """Load the trained policy for the given object with verification."""
        # An external controller supplies its own actions; nothing to load.
        if self.cfg.external_controller:
            self._current_policy = None
            self._current_vecnorm = None
            self._current_vecnorm_path = None
            return
        try:
            print(f"\n[POLICY] Loading policy for: {obj_info.object_id}")
            print(f"[POLICY] Policy file: {obj_info.policy_path}")
            
            # Verify policy file exists and matches object
            policy_path = Path(obj_info.policy_path)
            if not policy_path.exists():
                raise FileNotFoundError(f"Policy file not found: {policy_path}")
            
            # Check if policy path contains object ID (basic verification)
            if obj_info.object_id not in str(policy_path):
                print(f"[WARN] ⚠ Policy path does not contain object ID!")
                print(f"[WARN] Object: {obj_info.object_id}")
                print(f"[WARN] Policy: {policy_path}")
                print(f"[WARN] This might indicate a mismatch!")
            
            # Load policy
            self._current_policy = PPO.load(obj_info.policy_path, device=self.device)
            
            # Store vecnorm path
            self._current_vecnorm_path = obj_info.vecnorm_path if os.path.exists(obj_info.vecnorm_path) else None
            self._current_vecnorm = None
            
            if self._current_vecnorm_path:
                print(f"[POLICY] VecNormalize path: {self._current_vecnorm_path}")
            else:
                print(f"[POLICY] No VecNormalize found")
            
            print(f"[POLICY] ✓ Policy loaded successfully for {obj_info.object_id}\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to load policy for {obj_info.object_id}: {e}")
            import traceback
            traceback.print_exc()
            self._current_policy = None
            self._current_vecnorm = None
            self._current_vecnorm_path = None


    def _create_object_in_scene(self, obj_info: ObjectSpawnInfo):
        """Dynamically create an object in the scene."""
        stage = omni.usd.get_context().get_stage()
        
        # Remove old object from scene registry first
        if "object" in self.scene.rigid_objects:
            del self.scene.rigid_objects["object"]
        
        # Create object for each environment
        for env_idx in range(self.num_envs):
            env_path = self.scene.env_prim_paths[env_idx]
            object_path = f"{env_path}/Object"
            
            # Remove old object if exists
            old_prim = stage.GetPrimAtPath(object_path)
            if old_prim.IsValid():
                stage.RemovePrim(object_path)
            
            # Spawn new object
            object_cfg = RigidObjectCfg(
                prim_path=object_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=obj_info.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, self.cfg.table_height + 0.05),
                    rot=(1.0, 0.0, 0.0, 0.0)
                ),
            )
            
            # Spawn the object
            object_cfg.spawn.func(object_path, object_cfg.spawn)
        
        # Create NEW RigidObject wrapper with proper configuration
        object_cfg_full = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=obj_info.usd_path),
        )
        self.object = RigidObject(object_cfg_full)
        
        # Add to scene and initialize
        self.scene.rigid_objects["object"] = self.object
        
        # CRITICAL: Call reset to initialize internal state
        self.object.reset()
        
        # Force physics update
        for _ in range(10):
            self.sim.step(render=False)
        
        print(f"[INFO] Object {obj_info.object_id} spawned in scene")


    def _check_pick_success(self) -> bool:
        """Check if the current pick was successful."""
        if self.object is None:
            return False
        
        object_pos_world = self.object.data.root_pos_w[0]  # Single env
        object_init_pos_world = self.object_init_pos[0]
        
        height_change = object_pos_world[2] - object_init_pos_world[2]
        
        success = height_change >= (self.cfg.max_lift_height * 0.95)
        
        return success.item()


    def get_current_policy(self) -> Optional[PPO]:
        """Get the current active policy for external use."""
        return self._current_policy

    def get_current_vecnorm_path(self) -> Optional[str]:
        """Get the path to VecNormalize file for current policy."""
        return getattr(self, '_current_vecnorm_path', None)

    def _setup_scene(self):
        """Setup scene with robot and ALL objects in a cluttered arrangement."""

        self._contact_sensors_enabled = self.cfg.enable_contact_sensors
        
        # Create robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Create ALL objects that will be used in the experiment
        print(f"\n[INFO] Creating {len(self._object_infos)} objects in cluttered scene...")
        
        for i, obj_info in enumerate(self._object_infos):
            # Each object gets its own prim path
            object_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Object_{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=obj_info.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False,
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    # Will be positioned on table in _initialize_cluttered_scene
                    pos=(0.0, 0.0, self.cfg.table_height + 0.05),
                    rot=(1.0, 0.0, 0.0, 0.0)
                ),
            )
            
            obj = RigidObject(object_cfg)
            self.objects.append(obj)
            self.scene.rigid_objects[f"object_{i}"] = obj
            
            print(f"  Created object {i}: {obj_info.object_id}")

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Create table in source environment
        self._create_table_in_source()

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # ========== ADD CLASSIFIER CAMERA HERE ==========
        # CRITICAL: Add camera BEFORE cloning environments
        if self.cfg.enable_classifier_mode:
            print("\n[INFO] Adding classifier camera to scene...")
            from isaaclab.sensors import TiledCamera
            
            # Create camera instance
            self.classifier_camera = TiledCamera(self.cfg.classifier_camera)
            
            # Add to scene (this is the key step!)
            self.scene.sensors["classifier_camera"] = self.classifier_camera
            print("[INFO] Classifier camera added to scene")
        else:
            self.classifier_camera = None
        # ================================================
        
        print(f"[INFO] Scene setup complete with {len(self.objects)} objects")


    def _create_table_in_source(self):
        """Create table with legs in the source environment"""
        leg_height = self.cfg.table_height - self.cfg.table_thickness
        source_env_path = "/World/envs/env_0"
        
        # Table top
        table_top_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.table_width, self.cfg.table_depth, self.cfg.table_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        table_top_cfg.func(
            f"{source_env_path}/Table/TableTop",
            table_top_cfg,
            translation=(0.0, 0.0, self.cfg.table_height - self.cfg.table_thickness / 2)
        )
        
        # Table legs
        leg_offset_x = self.cfg.table_width / 2 - self.cfg.leg_radius - 0.02
        leg_offset_y = self.cfg.table_depth / 2 - self.cfg.leg_radius - 0.02
        
        leg_positions = [
            (leg_offset_x, leg_offset_y, leg_height / 2),
            (-leg_offset_x, leg_offset_y, leg_height / 2),
            (leg_offset_x, -leg_offset_y, leg_height / 2),
            (-leg_offset_x, -leg_offset_y, leg_height / 2),
        ]
        
        for i, pos in enumerate(leg_positions):
            leg_cfg = sim_utils.CylinderCfg(
                radius=self.cfg.leg_radius,
                height=leg_height,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=False,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            leg_cfg.func(f"{source_env_path}/Table/Leg{i}", leg_cfg, translation=pos)

    def _initialize_cluttered_scene(self):
        """Position all objects on table in a cluttered arrangement."""
        print(f"\n[INFO] Initializing cluttered scene with {len(self.objects)} objects...")
        
        num_objects = len(self.objects)
        
        # ============ CRITICAL SAFETY CHECK ============
        if num_objects != len(self._object_infos):
            raise RuntimeError(
                f"Object count mismatch! "
                f"self.objects has {num_objects} objects but "
                f"self._object_infos has {len(self._object_infos)} objects. "
                f"This should never happen!"
            )
        # ===============================================
        
        env_origins = self.scene.env_origins
        
        for attempt in range(self.cfg.max_spawn_attempts):
            print(f"\n[INFO] Spawn attempt {attempt + 1}/{self.cfg.max_spawn_attempts}")
            
            # Choose spawn method
            if self.cfg.use_clutter_based_spawn:
                print(f"[INFO] Using CLUTTER-BASED spawning for {self.cfg.target_complexity}")
                positions = self._generate_clutter_based_scene()
            elif self.cfg.use_predefined_scene:
                print(f"[INFO] Using PREDEFINED scene: {self.cfg.predefined_scene_name}")
                positions = self._get_predefined_scene_layout(
                    self.cfg.predefined_scene_name, 
                    num_objects
                )
            else:
                print(f"[INFO] Using RANDOM spawning")
                positions = self._generate_object_spawn_positions(num_objects)
            
            # ============ SECOND SAFETY CHECK ============
            print(f"[DEBUG] Generated {len(positions)} positions for {num_objects} objects")
            if len(positions) != num_objects:
                raise RuntimeError(
                    f"Position count mismatch! "
                    f"Generated {len(positions)} positions but need {num_objects}. "
                    f"Spawn method: {'clutter' if self.cfg.use_clutter_based_spawn else 'predefined' if self.cfg.use_predefined_scene else 'random'}"
                )
            # =============================================
            
            # Position objects
            for i, obj in enumerate(self.objects):
                obj_info = self._object_infos[i]
                pos_rel = positions[i]
                
                obj_pos_world = env_origins.clone()
                obj_pos_world[:, 0] += pos_rel[0]
                obj_pos_world[:, 1] += pos_rel[1]
                obj_pos_world[:, 2] += self.cfg.table_height + 0.05
                
                # Randomize orientation - ALWAYS randomize for all objects
                if self.cfg.randomize_object_orientation:
                    random_roll = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                    random_pitch = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                    random_yaw = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                    
                    random_euler = torch.stack([random_roll, random_pitch, random_yaw], dim=-1)
                    random_quat = euler_to_quaternion(random_euler)
                    random_quat = random_quat / torch.norm(random_quat, dim=-1, keepdim=True)
                else:
                    random_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
                
                object_state = obj.data.default_root_state.clone()
                object_state[:, 0:3] = obj_pos_world
                object_state[:, 3:7] = random_quat
                object_state[:, 7:] = 0.0
                
                obj.write_root_state_to_sim(object_state)
                
                print(f"  Placed {obj_info.object_id} at ({pos_rel[0]:.3f}, {pos_rel[1]:.3f})")
            
            # Let physics settle
            print(f"[INFO] Settling physics for {self.cfg.spawn_settling_steps} steps...")
            for _ in range(self.cfg.spawn_settling_steps):
                self.sim.step(render=False)
            
            # Validate spawn
            if self._validate_spawn_configuration():
                print("[INFO] ✓ Valid spawn configuration achieved!")
                break
            else:
                print(f"[WARN] ✗ Invalid spawn configuration, retrying...")
                if attempt == self.cfg.max_spawn_attempts - 1:
                    print("[ERROR] Failed to achieve valid spawn after max attempts!")
                    print("[ERROR] Proceeding with current configuration...")
        
        print("[INFO] Cluttered scene initialized!")
        
        # ============ CAPTURE TARGET INITIAL POSITION ============
        target_object = self.objects[0]
        target_object.update(dt=self.cfg.sim.dt)
        target_pos_world = target_object.data.root_pos_w[0].cpu()
        
        self.target_initial_pos = {
            0: (
                float(target_pos_world[0].item()),
                float(target_pos_world[1].item())
            )
        }
        
        # Also initialize last_table_pos (same as initial at spawn)
        self.target_last_table_pos = {
            0: (
                float(target_pos_world[0].item()),
                float(target_pos_world[1].item())
            )
        }
        
        # Initialize status
        self.target_status[0] = 'attempting'
        
        print(f"[CHAOS] Captured target initial position: ({self.target_initial_pos[0][0]:.3f}, {self.target_initial_pos[0][1]:.3f})")
        # =========================================================

    def _validate_spawn_configuration(self) -> bool:
        """Validate that all objects are in valid positions after settling.
        
        Checks:
        1. Objects are on the table (not fallen off)
        2. Objects are within reachable range of the hand
        3. Objects haven't rolled too far from their spawn positions
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        env_origins = self.scene.env_origins
        table_height = self.cfg.table_height
        
        all_valid = True
        
        for i, obj in enumerate(self.objects):
            obj_info = self._object_infos[i]
            obj_pos_world = obj.data.root_pos_w[0]  # Single env
            
            # Check 1: Object is on table (within tolerance)
            height_above_table = obj_pos_world[2] - (env_origins[0, 2] + table_height)
            if height_above_table < -self.cfg.spawn_height_tolerance or height_above_table > 0.1:
                print(f"  ✗ {obj_info.object_id}: Invalid height ({height_above_table:.3f}m above table)")
                all_valid = False
                continue
            
            # Check 2: Object is within reachable range (distance from table center)
            obj_pos_rel = obj_pos_world[:2] - env_origins[0, :2]  # XY only
            distance_from_center = torch.norm(obj_pos_rel).item()
            
            if distance_from_center > self.cfg.max_spawn_distance_from_origin:
                print(f"  ✗ {obj_info.object_id}: Out of reach ({distance_from_center:.3f}m from center)")
                all_valid = False
                continue
            
            # Check 3: Object is within table bounds
            table_x_limit = self.cfg.table_width / 2 - 0.05
            table_y_limit = self.cfg.table_depth / 2 - 0.05
            
            if abs(obj_pos_rel[0].item()) > table_x_limit or abs(obj_pos_rel[1].item()) > table_y_limit:
                print(f"  ✗ {obj_info.object_id}: Outside table bounds")
                all_valid = False
                continue
            
            print(f"  ✓ {obj_info.object_id}: Valid (dist={distance_from_center:.3f}m, height={height_above_table:.3f}m)")
        
        return all_valid

    def _initialize_site_transforms(self):
        """Query USD once to get local transforms of sites relative to parent bodies."""
        try:
            stage = omni.usd.get_context().get_stage()
            robot_prim_path = f"{self.scene.env_prim_paths[0]}/Robot/root"
            
            print(f"[INFO] Querying site transforms from: {robot_prim_path}")
            
            finger_local_pos = []
            finger_local_quat = []
            
            # Query finger site transforms
            for idx, (site_path, parent_name) in enumerate(zip(self.cfg.finger_site_usd_paths, self.cfg.finger_site_parent_bodies)):
                full_path = f"{robot_prim_path}/{site_path}"
                prim = stage.GetPrimAtPath(full_path)
                
                if prim.IsValid():
                    xformable = UsdGeom.Xformable(prim)
                    local_xform = xformable.GetLocalTransformation()
                    translation = local_xform.ExtractTranslation()
                    rotation = local_xform.ExtractRotationQuat()
                    
                    pos = [float(translation[0]), float(translation[1]), float(translation[2])]
                    quat = [
                        float(rotation.GetReal()),
                        float(rotation.GetImaginary()[0]),
                        float(rotation.GetImaginary()[1]),
                        float(rotation.GetImaginary()[2])
                    ]
                    
                    finger_local_pos.append(pos)
                    finger_local_quat.append(quat)
                else:
                    print(f"[WARN] Finger site not found: {full_path}")
                    finger_local_pos.append([0.0, 0.0, 0.0])
                    finger_local_quat.append([1.0, 0.0, 0.0, 0.0])
            
            # Query palm site transforms
            palm_local_pos = []
            palm_local_quat = []
            
            for idx, site_path in enumerate(self.cfg.palm_site_usd_paths):
                full_path = f"{robot_prim_path}/{site_path}"
                prim = stage.GetPrimAtPath(full_path)
                
                if prim.IsValid():
                    xformable = UsdGeom.Xformable(prim)
                    local_xform = xformable.GetLocalTransformation()
                    translation = local_xform.ExtractTranslation()
                    rotation = local_xform.ExtractRotationQuat()
                    
                    pos = [float(translation[0]), float(translation[1]), float(translation[2])]
                    quat = [
                        float(rotation.GetReal()),
                        float(rotation.GetImaginary()[0]),
                        float(rotation.GetImaginary()[1]),
                        float(rotation.GetImaginary()[2])
                    ]
                    
                    palm_local_pos.append(pos)
                    palm_local_quat.append(quat)
                else:
                    print(f"[WARN] Palm site not found: {full_path}")
                    palm_local_pos.append([0.0, 0.0, 0.0])
                    palm_local_quat.append([1.0, 0.0, 0.0, 0.0])
            
            # Convert to tensors
            self._finger_site_local_pos = torch.tensor(finger_local_pos, device=self.device, dtype=torch.float32)
            self._finger_site_local_quat = torch.tensor(finger_local_quat, device=self.device, dtype=torch.float32)
            self._palm_site_local_pos = torch.tensor(palm_local_pos, device=self.device, dtype=torch.float32)
            self._palm_site_local_quat = torch.tensor(palm_local_quat, device=self.device, dtype=torch.float32)
            
            self._sites_initialized = True
            print(f"[INFO] Successfully cached {len(finger_local_pos)} finger site transforms")
            print(f"[INFO] Successfully cached {len(palm_local_pos)} palm site transforms")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize site transforms: {e}")
            import traceback
            traceback.print_exc()
            # Initialize with zeros as fallback
            self._finger_site_local_pos = torch.zeros((10, 3), device=self.device)
            self._finger_site_local_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 10, device=self.device)
            self._palm_site_local_pos = torch.zeros((7, 3), device=self.device)
            self._palm_site_local_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 7, device=self.device)
            self._sites_initialized = False

    def _generate_picking_order(self):
        """Generate picking order - in single target mode, only pick object at index 0."""
        
        print(f"\n{'='*80}")
        print(f"[PICKING ORDER] Determining order")
        print(f"{'='*80}")
        
        if self.cfg.single_object_mode:
            # ============ SINGLE TARGET MODE ============
            # Target object MUST be at index 0 in self.objects
            # Picking order is simply [0]
            self._picking_order = [0]
            
            target_obj_name = self._object_infos[0].object_id
            
            print(f"[PICKING ORDER] SINGLE TARGET MODE")
            print(f"[PICKING ORDER] Target object at index 0: {target_obj_name}")
            print(f"[PICKING ORDER] Picking order: {self._picking_order}")
            print(f"[PICKING ORDER] Other {len(self.objects) - 1} objects are static clutter")
            
            # Verify target is correct
            if self.cfg.target_object_id and target_obj_name != self.cfg.target_object_id:
                print(f"[ERROR] ⚠️⚠️⚠️ MISMATCH DETECTED!")
                print(f"[ERROR] Expected target: {self.cfg.target_object_id}")
                print(f"[ERROR] Object at index 0: {target_obj_name}")
                print(f"[ERROR] This should not happen!")
        else:
            # ============ MULTI-OBJECT MODE ============
            num_objects = len(self.objects)
            self._picking_order = list(np.random.permutation(num_objects))
            print(f"[PICKING ORDER] MULTI-OBJECT MODE - randomized order:")
            for pick_idx, obj_idx in enumerate(self._picking_order):
                obj_id = self._object_infos[obj_idx].object_id
                print(f"  {pick_idx + 1}. {obj_id} (object index: {obj_idx})")
        
        print(f"{'='*80}\n")
        
        self._current_object_idx = -1

    def _generate_object_spawn_positions(self, num_objects: int) -> list:
        """Generate spawn positions for objects on table.
        
        Returns list of (x, y) positions relative to table center.
        """
        
        # ============ ISOLATED MODE ============
        if self.cfg.use_isolated_mode:
            print("[INFO] ISOLATED MODE: Random position for single object")
            x_min, x_max = self.cfg.isolated_spawn_area_x
            y_min, y_max = self.cfg.isolated_spawn_area_y
            
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            print(f"[INFO] Isolated object position: ({x:.3f}, {y:.3f})")
            return [(x, y)]
        # =======================================
        
        # ============ CHECK FOR PREDEFINED SCENE ============
        if self.cfg.use_predefined_scene:
            print(f"[INFO] Using predefined scene: {self.cfg.predefined_scene_name}")
            return self._get_predefined_scene_layout(self.cfg.predefined_scene_name, num_objects)
        # ====================================================
        
        positions = []
        
        # Calculate usable table area (with margins)
        margin = self.cfg.spawn_area_margin
        usable_width = self.cfg.table_width - 2 * margin
        usable_depth = self.cfg.table_depth - 2 * margin
        
        if self.cfg.randomize_spawn_positions:
            # Random placement with collision avoidance
            print("[INFO] Generating random spawn positions with collision avoidance...")
            positions = self._generate_random_positions(num_objects)
        else:
            # Grid placement
            print("[INFO] Generating grid spawn positions...")
            for i in range(num_objects):
                pos = self._get_grid_position(i, num_objects, usable_width, usable_depth)
                positions.append(pos)
        
        return positions

    def _generate_clutter_based_scene(self):
        """Generate scene with target object at center and complexity-controlled clutter.
        
        CRITICAL: Must return EXACTLY len(self._object_infos) positions
        """
        from .benchmark_env_cfg import CLUTTER_CONFIGS
        
        if self.cfg.target_complexity not in CLUTTER_CONFIGS:
            raise ValueError(f"Invalid target_complexity: {self.cfg.target_complexity}")
        
        clutter_config = CLUTTER_CONFIGS[self.cfg.target_complexity]
        
        print(f"\n{'='*80}")
        print(f"[CLUTTER SPAWN] Generating {self.cfg.target_complexity} scene")
        print(f"[CLUTTER SPAWN] {clutter_config['description']}")
        print(f"[CLUTTER SPAWN] Need positions for {len(self._object_infos)} total objects")
        print(f"{'='*80}")
        
        # Determine actual number of neighbors from object count
        num_total_objects = len(self._object_infos)
        num_neighbors = num_total_objects - 1  # Minus 1 for target
        
        # Check if we have far objects (more objects than max neighbors + 1)
        max_neighbors = clutter_config['num_neighbors'][1]
        if num_neighbors > max_neighbors:
            num_close_neighbors = max_neighbors
            num_far_objects = num_neighbors - max_neighbors
            print(f"[CLUTTER SPAWN] Close neighbors: {num_close_neighbors}")
            print(f"[CLUTTER SPAWN] Far objects: {num_far_objects}")
        else:
            num_close_neighbors = num_neighbors
            num_far_objects = 0
            print(f"[CLUTTER SPAWN] Close neighbors: {num_close_neighbors}")
        
        positions = []
        
        # ============ STEP 1: Target at Center ============
        target_pos = (0.0, 0.0)
        positions.append(target_pos)
        print(f"[CLUTTER SPAWN] Target position: (0.000, 0.000)")
        
        # ============ STEP 2: Spawn Close Neighbors ============
        if num_close_neighbors > 0:
            neighbor_positions = self._generate_neighbor_positions(
                target_pos,
                num_close_neighbors,
                clutter_config['min_clearance'],
                clutter_config['max_clearance']
            )
            
            positions.extend(neighbor_positions)
            
            print(f"[CLUTTER SPAWN] Spawned {len(neighbor_positions)} close neighbors:")
            for i, pos in enumerate(neighbor_positions):
                dist = np.sqrt(pos[0]**2 + pos[1]**2)
                print(f"  Neighbor {i+1}: ({pos[0]:+.3f}, {pos[1]:+.3f}) - {dist:.3f}m from target")
        
        # ============ STEP 3: Add Distant Objects ============
        if num_far_objects > 0:
            print(f"[CLUTTER SPAWN] Adding {num_far_objects} distant objects...")
            
            far_positions = self._generate_far_object_positions(
                positions,
                num_far_objects,
                min_distance=0.25
            )
            
            positions.extend(far_positions)
            print(f"[CLUTTER SPAWN] Added {len(far_positions)} distant objects")
        
        # ============ CRITICAL VERIFICATION ============
        if len(positions) != num_total_objects:
            raise RuntimeError(
                f"CLUTTER SPAWN BUG: Generated {len(positions)} positions but need {num_total_objects}!"
            )
        
        print(f"[CLUTTER SPAWN] Total positions generated: {len(positions)}")
        print(f"{'='*80}\n")
        
        return positions


    def _generate_neighbor_positions(self, center_pos: tuple, num_neighbors: int,
                                    min_clearance: float, max_clearance: float) -> list:
        """Generate positions for neighboring objects around target.
        
        Args:
            center_pos: (x, y) position of target object
            num_neighbors: Number of neighbors to place
            min_clearance: Minimum distance from target
            max_clearance: Maximum distance from target
        
        Returns:
            List of (x, y) positions for neighbors
        """
        positions = []
        
        # Generate positions in a ring around target
        for i in range(num_neighbors):
            # Distribute evenly around target with some randomness
            base_angle = (i / num_neighbors) * 2 * np.pi
            angle_noise = np.random.uniform(-0.3, 0.3)  # ±17 degrees
            angle = base_angle + angle_noise
            
            # Random distance within clearance range
            distance = np.random.uniform(min_clearance, max_clearance)
            
            # Calculate position
            x = center_pos[0] + distance * np.cos(angle)
            y = center_pos[1] + distance * np.sin(angle)
            
            # Ensure within table bounds
            margin = 0.15
            table_x_limit = self.cfg.table_width / 2 - margin
            table_y_limit = self.cfg.table_depth / 2 - margin
            
            x = np.clip(x, -table_x_limit, table_x_limit)
            y = np.clip(y, -table_y_limit, table_y_limit)
            
            positions.append((x, y))
        
        return positions


    def _generate_far_object_positions(self, existing_positions: list, num_far: int,
                                    min_distance: float = 0.25) -> list:
        """Generate positions for distant objects that don't interfere with target area.
        
        Args:
            existing_positions: List of (x, y) positions already occupied
            num_far: Number of far objects to place
            min_distance: Minimum distance from any existing object
        
        Returns:
            List of (x, y) positions for far objects
        """
        positions = []
        margin = 0.15
        table_x_limit = self.cfg.table_width / 2 - margin
        table_y_limit = self.cfg.table_depth / 2 - margin
        
        max_attempts = 100
        
        for _ in range(num_far):
            placed = False
            
            for attempt in range(max_attempts):
                # Random position on table
                x = np.random.uniform(-table_x_limit, table_x_limit)
                y = np.random.uniform(-table_y_limit, table_y_limit)
                
                # Check distance from all existing objects
                min_dist = min([
                    np.sqrt((x - ex)**2 + (y - ey)**2)
                    for ex, ey in existing_positions + positions
                ])
                
                if min_dist >= min_distance:
                    positions.append((x, y))
                    placed = True
                    break
            
            if not placed:
                print(f"[WARN] Could not place far object {len(positions)+1}, max attempts reached")
        
        return positions

    def _get_predefined_scene_layout(self, scene_name: str, num_objects: int) -> list:
        """Generate predefined scene layouts with specific clustering patterns.
        
        DETERMINISTIC: Always returns the same positions for the same scene.
        
        Args:
            scene_name: Name of the predefined scene ("scene1", "scene2", "scene3")
            num_objects: Total number of objects to place
            
        Returns:
            List of (x, y) positions relative to table center
        """
        positions = []
        
        # Calculate usable table area (with margins)
        margin = self.cfg.spawn_area_margin
        usable_width = self.cfg.table_width - 2 * margin
        usable_depth = self.cfg.table_depth - 2 * margin
        
        print(f"\n{'='*80}")
        print(f"[SCENE LAYOUT] Creating DETERMINISTIC predefined scene: {scene_name}")
        print(f"{'='*80}")
        
        if scene_name == "scene1":
            # Scene 1: 10 objects
            # - 3 objects densely clustered (spacing ~2cm)
            # - 3 objects moderately clustered (spacing ~5cm)
            # - 4 objects scattered
            
            print("[SCENE1] Layout: 3 dense + 3 moderate + 4 scattered (DETERMINISTIC)")
            
            # Dense cluster (top-left quadrant) - FIXED POSITIONS
            dense_center = (-usable_width * 0.25, usable_depth * 0.25)
            dense_spacing = 0.025  # 2.5cm spacing
            dense_positions = [
                (dense_center[0], dense_center[1]),
                (dense_center[0] + dense_spacing, dense_center[1]),
                (dense_center[0] + dense_spacing/2, dense_center[1] + dense_spacing),
            ]
            positions.extend(dense_positions[:min(3, num_objects)])
            print(f"  Dense cluster at {dense_center}: 3 objects")
            
            # Moderate cluster (top-right quadrant) - FIXED POSITIONS
            if num_objects > 3:
                moderate_center = (usable_width * 0.25, usable_depth * 0.25)
                moderate_spacing = 0.055  # 5.5cm spacing
                moderate_positions = [
                    (moderate_center[0], moderate_center[1]),
                    (moderate_center[0] + moderate_spacing, moderate_center[1]),
                    (moderate_center[0] + moderate_spacing/2, moderate_center[1] + moderate_spacing),
                ]
                positions.extend(moderate_positions[:min(3, num_objects - len(positions))])
                print(f"  Moderate cluster at {moderate_center}: 3 objects")
            
            # Scattered objects (bottom half + center) - FIXED POSITIONS
            if num_objects > 6:
                scattered_positions = [
                    (-usable_width * 0.3, -usable_depth * 0.2),
                    (usable_width * 0.3, -usable_depth * 0.3),
                    (0.0, 0.0),  # Center
                    (-usable_width * 0.15, -usable_depth * 0.35),
                ]
                positions.extend(scattered_positions[:num_objects - len(positions)])
                print(f"  Scattered: {num_objects - 6} objects (FIXED positions)")
        
        elif scene_name == "scene2":
            # Scene 2: 15 objects
            # - 8 objects in moderate cluster at center (~5.5cm spacing)
            # - 7 objects scattered far around table perimeter
            
            print("[SCENE2] Layout: 8 moderate center cluster + 7 far scattered (DETERMINISTIC)")
            
            # Moderate cluster (center, 8 objects) - FIXED POSITIONS
            moderate_spacing = 0.055  # 5.5cm spacing
            cluster_center = (0.0, 0.0)
            
            # Create moderate cluster pattern (compact 8-object arrangement) - FIXED POSITIONS
            cluster_positions = [
                # Center object
                (cluster_center[0], cluster_center[1]),
                # Inner ring (4 objects around center)
                (cluster_center[0] + moderate_spacing, cluster_center[1]),                    # Right
                (cluster_center[0] - moderate_spacing, cluster_center[1]),                    # Left
                (cluster_center[0], cluster_center[1] + moderate_spacing),                    # Top
                (cluster_center[0], cluster_center[1] - moderate_spacing),                    # Bottom
                # Diagonal positions (3 more objects to make 8 total)
                (cluster_center[0] + moderate_spacing, cluster_center[1] + moderate_spacing), # Top-right
                (cluster_center[0] - moderate_spacing, cluster_center[1] + moderate_spacing), # Top-left
                (cluster_center[0] + moderate_spacing, cluster_center[1] - moderate_spacing), # Bottom-right
            ]
            positions.extend(cluster_positions[:min(8, num_objects)])
            print(f"  Moderate cluster at center: 8 objects (FIXED positions)")
            
            # Scattered objects around perimeter (far from cluster) - FIXED POSITIONS
            if num_objects > 8:
                # Place objects near table edges at fixed positions
                # Using larger distances from center (around 0.30-0.35m)
                far_scattered_positions = [
                    (-usable_width * 0.40, -usable_depth * 0.35),  # Bottom-left corner
                    (usable_width * 0.40, -usable_depth * 0.35),   # Bottom-right corner
                    (-usable_width * 0.40, usable_depth * 0.35),   # Top-left corner
                    (usable_width * 0.40, usable_depth * 0.35),    # Top-right corner
                    (0.0, -usable_depth * 0.40),                   # Bottom edge (center)
                    (0.0, usable_depth * 0.40),                    # Top edge (center)
                    (-usable_width * 0.40, 0.0),                   # Left edge (center)
                ]
                positions.extend(far_scattered_positions[:num_objects - len(positions)])
                print(f"  Far scattered: {min(7, num_objects - 8)} objects around perimeter (FIXED positions)")
        
        elif scene_name == "scene3":
            # Scene 3: 20 objects
            # - 2 super dense clusters with 5 objects each (~2cm spacing)
            # - 1 moderate cluster with 6 objects (~5cm spacing)
            # - Remaining scattered around table
            
            print("[SCENE3] Layout: 2 super dense (5 each) + 1 moderate (6) + scattered (DETERMINISTIC)")
            
            # Super dense cluster 1 (left side) - FIXED POSITIONS
            dense1_center = (-usable_width * 0.3, usable_depth * 0.25)
            dense_spacing = 0.025  # 2.5cm
            dense1_positions = [
                (dense1_center[0], dense1_center[1]),
                (dense1_center[0] + dense_spacing, dense1_center[1]),
                (dense1_center[0] - dense_spacing, dense1_center[1]),
                (dense1_center[0], dense1_center[1] + dense_spacing),
                (dense1_center[0], dense1_center[1] - dense_spacing),
            ]
            positions.extend(dense1_positions[:min(5, num_objects)])
            print(f"  Super dense cluster 1 at {dense1_center}: 5 objects (FIXED positions)")
            
            # Super dense cluster 2 (right side) - FIXED POSITIONS
            if num_objects > 5:
                dense2_center = (usable_width * 0.3, usable_depth * 0.25)
                dense2_positions = [
                    (dense2_center[0], dense2_center[1]),
                    (dense2_center[0] + dense_spacing, dense2_center[1]),
                    (dense2_center[0] - dense_spacing, dense2_center[1]),
                    (dense2_center[0], dense2_center[1] + dense_spacing),
                    (dense2_center[0], dense2_center[1] - dense_spacing),
                ]
                positions.extend(dense2_positions[:min(5, num_objects - len(positions))])
                print(f"  Super dense cluster 2 at {dense2_center}: 5 objects (FIXED positions)")
            
            # Moderate cluster (center-bottom) - FIXED POSITIONS
            if num_objects > 10:
                moderate_center = (0.0, -usable_depth * 0.15)
                moderate_spacing = 0.055  # 5.5cm
                moderate_positions = [
                    (moderate_center[0], moderate_center[1]),
                    (moderate_center[0] + moderate_spacing, moderate_center[1]),
                    (moderate_center[0] - moderate_spacing, moderate_center[1]),
                    (moderate_center[0] + moderate_spacing/2, moderate_center[1] + moderate_spacing),
                    (moderate_center[0] - moderate_spacing/2, moderate_center[1] + moderate_spacing),
                    (moderate_center[0], moderate_center[1] - moderate_spacing),
                ]
                positions.extend(moderate_positions[:min(6, num_objects - len(positions))])
                print(f"  Moderate cluster at {moderate_center}: 6 objects (FIXED positions)")
            
            # Scatter remaining objects - DETERMINISTIC POSITIONS
            if num_objects > 16:
                num_scattered = num_objects - len(positions)
                print(f"  Scattering {num_scattered} remaining objects (DETERMINISTIC positions)...")
                
                # DETERMINISTIC: Use fixed angles and radii (no randomization)
                for i in range(num_scattered):
                    # Fixed angle based on index
                    angle = (i / num_scattered) * 2 * np.pi
                    # Fixed radius pattern
                    radius = 0.20 + (i % 2) * 0.08  # Alternate between 0.20 and 0.28
                    
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    
                    # Clamp to bounds
                    x = np.clip(x, -usable_width/2 + 0.05, usable_width/2 - 0.05)
                    y = np.clip(y, -usable_depth/2 + 0.05, usable_depth/2 - 0.05)
                    
                    positions.append((x, y))
        
        else:
            print(f"[ERROR] Unknown predefined scene: {scene_name}")
            print(f"[ERROR] Falling back to grid placement")
            return self._generate_grid_positions(num_objects)
        
        # Ensure we have exactly num_objects positions
        if len(positions) < num_objects:
            print(f"[WARN] Only generated {len(positions)}/{num_objects} positions")
            print(f"[WARN] Filling remaining with grid positions...")
            # Use grid for remaining (deterministic)
            usable_width = self.cfg.table_width - 2 * self.cfg.spawn_area_margin
            usable_depth = self.cfg.table_depth - 2 * self.cfg.spawn_area_margin
            for i in range(len(positions), num_objects):
                grid_pos = self._get_grid_position(i, num_objects, usable_width, usable_depth)
                positions.append(grid_pos)
        elif len(positions) > num_objects:
            positions = positions[:num_objects]
        
        print(f"[SCENE LAYOUT] ✓ Generated {len(positions)} DETERMINISTIC positions")
        print(f"{'='*80}\n")
        
        return positions
    
    def _generate_random_clutter_positions(self, num_clutter: int) -> list:
        """Generate random positions for clutter objects (avoiding center).
        
        Args:
            num_clutter: Number of clutter objects to place
        
        Returns:
            List of (x, y) positions relative to table center
        """
        positions = []
        
        margin = self.cfg.spawn_area_margin
        usable_width = self.cfg.table_width - 2 * margin
        usable_depth = self.cfg.table_depth - 2 * margin
        
        # Minimum distance from center (to avoid overlap with target)
        min_dist_from_center = 0.08  # 8cm clearance from center
        
        for i in range(num_clutter):
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                # Random position
                x = (np.random.rand() - 0.5) * usable_width
                y = (np.random.rand() - 0.5) * usable_depth
                
                # Check distance from center
                dist_from_center = np.sqrt(x**2 + y**2)
                if dist_from_center < min_dist_from_center:
                    continue  # Too close to center target
                
                # Check distance from other clutter
                if len(positions) == 0:
                    positions.append((x, y))
                    placed = True
                    break
                
                min_dist = min([
                    np.sqrt((x - px)**2 + (y - py)**2) 
                    for px, py in positions
                ])
                
                if min_dist >= self.cfg.min_object_spacing:
                    positions.append((x, y))
                    placed = True
                    break
            
            if not placed:
                # Fallback: place at edge
                angle = (i / num_clutter) * 2 * np.pi
                radius = usable_width / 3
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append((x, y))
        
        return positions

    def _generate_random_positions(self, num_objects: int) -> list:
        """Generate random positions with collision avoidance (helper method)."""
        positions = []
        margin = self.cfg.spawn_area_margin
        usable_width = self.cfg.table_width - 2 * margin
        usable_depth = self.cfg.table_depth - 2 * margin
        
        for i in range(num_objects):
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                x = (torch.rand(1).item() - 0.5) * usable_width
                y = (torch.rand(1).item() - 0.5) * usable_depth
                
                if len(positions) == 0:
                    positions.append((x, y))
                    placed = True
                    break
                
                min_dist = min([
                    np.sqrt((x - px)**2 + (y - py)**2) 
                    for px, py in positions
                ])
                
                if min_dist >= self.cfg.min_object_spacing:
                    positions.append((x, y))
                    placed = True
                    break
            
            if not placed:
                # Fallback to grid if random fails
                grid_pos = self._get_grid_position(i, num_objects, usable_width, usable_depth)
                positions.append(grid_pos)
        
        return positions

    def _generate_grid_positions(self, num_objects: int) -> list:
        """Generate DETERMINISTIC grid positions (helper for fallback)."""
        positions = []
        margin = self.cfg.spawn_area_margin
        usable_width = self.cfg.table_width - 2 * margin
        usable_depth = self.cfg.table_depth - 2 * margin
        
        for i in range(num_objects):
            pos = self._get_grid_position(i, num_objects, usable_width, usable_depth)
            positions.append(pos)
        
        return positions

    def _get_grid_position(self, index: int, total: int, width: float, depth: float) -> tuple:
        """Get grid position for object placement."""
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(total)))
        rows = int(np.ceil(total / cols))
        
        col = index % cols
        row = index // cols
        
        # Calculate spacing
        x_spacing = width / (cols + 1)
        y_spacing = depth / (rows + 1)
        
        # Calculate position (centered in grid)
        x = (col + 1) * x_spacing - width / 2
        y = (row + 1) * y_spacing - depth / 2
        
        return (x, y)


    def _check_for_knocked_out_objects(self):
        """Check if any non-target objects have been knocked off the table.
        
        This runs EVERY step to detect objects that fall off due to collisions
        with the hand or other objects during picking attempts.
        """
        env_origins = self.scene.env_origins
        table_surface_world = env_origins[:, 2] + self.cfg.table_height
        
        for obj_idx in range(len(self.objects)):
            # Skip if already processed
            if self._object_final_status.get(obj_idx) in ['success', 'knocked_out']:
                continue
            
            # Skip current target (handled separately in _get_dones)
            if self._current_object_idx < len(self._picking_order):
                if obj_idx == self._picking_order[self._current_object_idx]:
                    continue
            
            # Check if object fell off table
            obj = self.objects[obj_idx]
            obj.update(dt=self.cfg.sim.dt)
            obj_pos_world = obj.data.root_pos_w[0]
            
            if obj_pos_world[2] < (table_surface_world[0] - self.cfg.object_fall_margin):
                obj_info = self._object_infos[obj_idx]
                print(f"[KNOCKED OUT] Object {obj_idx} ({obj_info.object_id}) fell off table (non-target)")
                
                # Mark as knocked out
                self._object_final_status[obj_idx] = 'knocked_out'
                self._object_drop_counts[obj_idx] = self._object_drop_counts.get(obj_idx, 0)


    def _check_hand_stuck(self) -> bool:
        """Detect if hand is stuck by monitoring policy action outputs.
        
        If the policy outputs nearly identical actions repeatedly, it suggests:
        - Policy is saturated (hitting limits)
        - Hand is stuck in collision
        - Policy is converged to a bad local minimum
        
        Returns:
            bool: True if hand appears stuck and should be reset
        """
        current_step = self.common_step_counter
        
        # Only check periodically
        if current_step - self._last_hand_check_step < self._hand_stuck_check_interval:
            return False
        
        self._last_hand_check_step = current_step
        
        # Get current action
        if not hasattr(self, 'actions') or self.actions is None:
            return False
        
        current_action = self.actions[0].clone()
        
        # Store in history (keep last 5 actions)
        self._action_history.append(current_action)
        if len(self._action_history) > 5:
            self._action_history.pop(0)
        
        # Need at least 5 actions to check
        if len(self._action_history) < 5:
            return False
        
        # Stack actions into tensor (5, 12)
        action_stack = torch.stack(self._action_history, dim=0)
        
        # Calculate variance across time for each action dimension
        action_variance = torch.var(action_stack, dim=0)  # (12,)
        mean_variance = action_variance.mean().item()
        
        # Check if actions are nearly constant (low variance = stuck)
        if mean_variance < self._hand_stuck_action_threshold:
            return True
        
        return False
    
    def _check_hand_flip(self) -> bool:
        """Detect if hand has flipped upside-down without lifting object.
        
        Returns:
            bool: True if hand is flipped and object not lifted
        """
        if not self.cfg.detect_hand_flip:
            return False
        
        # Get current hand orientation
        current_euler = self.current_euler[0]  # Single env
        
        # Compute deviation from reference orientation
        euler_diff = wrap_angle_diff_for_limits(
            current_euler.unsqueeze(0), 
            self.ref_palm_euler[0:1]
        )[0]
        
        # Check if roll or pitch exceeded flip threshold
        roll_flipped = torch.abs(euler_diff[0]) > self.cfg.hand_flip_threshold
        pitch_flipped = torch.abs(euler_diff[1]) > self.cfg.hand_flip_threshold
        
        hand_is_flipped = roll_flipped or pitch_flipped
        
        # Only trigger if object is NOT lifted
        if hand_is_flipped:
            obj_idx = self._picking_order[self._current_object_idx]
            current_object = self.objects[obj_idx]
            current_object.update(dt=self.cfg.sim.dt)
            
            object_pos_world = current_object.data.root_pos_w[0]
            object_init_pos_world = self.object_init_pos[0]
            height_change = object_pos_world[2] - object_init_pos_world[2]
            
            object_is_lifted = height_change >= self.cfg.min_lift_height
            
            if not object_is_lifted:
                print(f"[HAND FLIP] Detected! Roll: {euler_diff[0]:.2f}rad, Pitch: {euler_diff[1]:.2f}rad")
                return True
        
        return False
    
    def _check_trial_step_limit(self) -> bool:
        """Check if current trial has exceeded step limit.
        
        Returns:
            bool: True if trial should respawn hand
        """
        if self._trial_step_counter >= self.cfg.max_steps_per_trial:
            print(f"[TRIAL LIMIT] Reached {self._trial_step_counter} steps, respawning hand")
            return True
        return False
    
    def _capture_and_annotate_target_for_verification(self):
        """Capture scene image and draw green box on target for classifier verification.
        
        Returns:
            str: Path to annotated image, or None if failed
        """
        import cv2
        from pathlib import Path
        from PIL import Image
        
        if self.classifier_camera is None:
            return None
        
        # Render and capture
        for i in range(20):
            self.sim.step(render=True)
            if i % 5 == 0:
                self.classifier_camera.update(dt=self.cfg.sim.dt)
        
        self.classifier_camera.update(dt=self.cfg.sim.dt)
        
        # Get camera data
        camera_data = self.classifier_camera.data
        cam_pos = camera_data.pos_w[0].clone()
        cam_quat = camera_data.quat_w_ros[0].clone()
        
        # Get RGB image
        if "rgb" not in self.classifier_camera.data.output:
            print(f"[ERROR] No RGB data in camera output")
            return None
        
        rgb_data = self.classifier_camera.data.output["rgb"][0]
        rgb_np = rgb_data.cpu().numpy()
        
        if rgb_np.dtype != np.uint8:
            rgb_uint8 = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_np
        
        # Extract target point cloud
        target_obj_info = self._object_infos[0]
        
        mesh_points_world = extract_policy_pointcloud_for_object(self, 0, target_obj_info)
        
        if mesh_points_world is None or len(mesh_points_world) == 0:
            print(f"[ERROR] Could not extract point cloud for target")
            return None
        
        # Project to image
        projected_2d = project_points_to_image_cached(
            mesh_points_world, cam_pos, cam_quat, 
            self.cfg.classifier_camera
        )
        
        if projected_2d is None:
            print(f"[ERROR] Could not project target to image")
            return None
        
        xs, ys = projected_2d
        
        width = self.cfg.classifier_camera.width
        height = self.cfg.classifier_camera.height
        
        padding = 2
        x_min = max(0, int(xs.min()) - padding)
        x_max = min(width - 1, int(xs.max()) + padding)
        y_min = max(0, int(ys.min()) - padding)
        y_max = min(height - 1, int(ys.max()) + padding)
        
        # Draw GREEN bounding box on target
        annotated_img = rgb_uint8.copy()
        cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), 
                    (50, 255, 50), 2)  # GREEN box
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp_verification_{timestamp}.png"
        filepath = Path(self.cfg.classifier_image_dir) / filename
        
        Image.fromarray(annotated_img).save(str(filepath))
        
        return str(filepath)
    
    def _respawn_object_for_new_trial(self):
        """Respawn scene for next trial:
        - Same target object at SAME position (deterministic)
        - Different random clutter objects (from available pool)
        - Different number of clutter objects
        - Capture FRESH annotated image each trial with consistent style
        - Proper chaos tracking with position capture
        - CLASSIFIER RUNS ON EVERY TRIAL (FIXED)
        """
        import cv2
        import os
        from pathlib import Path
        from PIL import Image
        
        obj_idx = 0  # Target object (always index 0)
        target_object = self.objects[obj_idx]
        obj_info = self._object_infos[obj_idx]
        
        print(f"\n{'='*80}")
        print(f"[TRIAL {self._current_trial + 1}/{self.cfg.num_trials}] Respawning scene")
        print(f"{'='*80}")

        # ============ RANDOMIZE WHICH OBJECTS TO USE ============
        print(f"[TRIAL] Randomizing object types from pool...")
        
        # Keep target (index 0) the same
        target_obj_info = self._object_infos[0]
        
        # Get available objects excluding target
        available_for_clutter = [obj for obj in self.cfg.available_objects 
                                if obj.object_id != target_obj_info.object_id]
        
        if len(available_for_clutter) > 0:
            # Calculate slots needed
            num_clutter_slots = len(self._object_infos) - 1
            
            # RANDOMLY select new object types
            new_clutter_indices = np.random.choice(
                len(available_for_clutter),
                size=num_clutter_slots,
                replace=True  # Allow duplicates
            )
            
            # Build new list
            new_object_infos = [target_obj_info]
            for idx in new_clutter_indices:
                new_object_infos.append(available_for_clutter[idx])
            
            self._object_infos = new_object_infos
            
            print(f"[TRIAL] Selected NEW object types:")
            for i in range(1, min(len(self._object_infos), 6)):
                print(f"  Slot {i}: {self._object_infos[i].object_id}")
        
        print(f"[TRIAL] ✓ Object types randomized")
        
        # ============ ISOLATED MODE: SAME position for target ============
        if self.cfg.use_isolated_mode:
            # Target at SAME position (center)
            x = 0.0
            y = 0.0
            
            print(f"[TRIAL] Mode: Isolated")
            print(f"[TRIAL] Target position (SAME): ({x:.4f}, {y:.4f})")
            
            env_origins = self.scene.env_origins
            
            obj_pos_world = env_origins.clone()
            obj_pos_world[:, 0] += x
            obj_pos_world[:, 1] += y
            obj_pos_world[:, 2] += self.cfg.table_height + 0.05
            
            # Randomize orientation
            if self.cfg.randomize_object_orientation:
                random_roll = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                random_pitch = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                random_yaw = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                
                random_euler = torch.stack([random_roll, random_pitch, random_yaw], dim=-1)
                random_quat = euler_to_quaternion(random_euler)
                random_quat = random_quat / torch.norm(random_quat, dim=-1, keepdim=True)
            else:
                random_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
            
            object_state = target_object.data.default_root_state.clone()
            object_state[:, 0:3] = obj_pos_world
            object_state[:, 3:7] = random_quat
            object_state[:, 7:] = 0.0
            
            target_object.write_root_state_to_sim(object_state)
            
            # Physics settling
            print(f"[TRIAL] Settling object...")
            for _ in range(50):
                self.sim.step(render=False)
            
            target_object.update(dt=self.cfg.sim.dt)
            
            for _ in range(30):
                self.sim.step(render=False)
            
            target_object.update(dt=self.cfg.sim.dt)
            
            # ── Lock in the SETTLED position as object_init_pos ───────────────
            # This must happen here (after physics settle) so the policy obs
            # object_init_pos_rel matches where the object actually rests.
            # Without this, object_init_pos still holds the pre-settle spawn
            # height (table_height + 0.05), causing a mismatch that makes the
            # policy move the hand away from the object in isolated mode.
            settled_pos = target_object.data.root_pos_w.clone()
            self.object_init_pos = settled_pos
            print(f"[TRIAL] Isolated: settled object_init_pos updated to "
                  f"({settled_pos[0,0]:.4f}, {settled_pos[0,1]:.4f}, {settled_pos[0,2]:.4f})")
            
            # Capture image (simple for isolated mode - just target with green box)
            if self.cfg.enable_classifier_mode and self.classifier_camera is not None:
                print(f"[TRIAL] Capturing scene image...")
                
                for i in range(20):
                    self.sim.step(render=True)
                    if i % 5 == 0:
                        self.classifier_camera.update(dt=self.cfg.sim.dt)
                
                self.classifier_camera.update(dt=self.cfg.sim.dt)
                
                if "rgb" in self.classifier_camera.data.output:
                    rgb_data = self.classifier_camera.data.output["rgb"][0]
                    rgb_np = rgb_data.cpu().numpy()
                    
                    if rgb_np.dtype != np.uint8:
                        base_image = (rgb_np * 255).astype(np.uint8)
                    else:
                        base_image = rgb_np
                    
                    # Add trial labels
                    trial_text = f"Trial {self._current_trial + 1}/{self.cfg.num_trials}"
                    mode_text = "Mode: Isolated"
                    object_text = f"Object: {obj_info.object_id}"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    color = (0, 255, 0)
                    
                    cv2.putText(base_image, trial_text, (10, 40), font, font_scale, color, thickness)
                    cv2.putText(base_image, mode_text, (10, 80), font, font_scale, color, thickness)
                    cv2.putText(base_image, object_text, (10, 120), font, font_scale, color, thickness)
                    
                    # Save
                    new_name = f"trial_{self._current_trial + 1:02d}_isolated_{obj_info.object_id}.png"
                    new_path = Path(self.cfg.classifier_image_dir) / new_name
                    cv2.imwrite(str(new_path), base_image)
                    
                    print(f"[TRIAL] ✓ Image saved: {new_name}")
        
        # ============ CLUTTER MODE: SAME target, RANDOMIZED clutter ============
        else:
            print(f"[TRIAL] Mode: Clutter - Target SAME, clutter RANDOMIZED")
            
            max_attempts = self.cfg.max_spawn_attempts if hasattr(self.cfg, 'max_spawn_attempts') else 10
            verified = False
            
            for attempt in range(max_attempts):
                print(f"\n[TRIAL] Spawn attempt {attempt + 1}/{max_attempts}")
                
                # ========== 1. REMOVE HAND ==========
                safe_pos = self.scene.env_origins.clone()
                safe_pos[:, 0] = 10.0
                safe_pos[:, 1] = 10.0
                safe_pos[:, 2] = -5.0
                
                hand_state = self.robot.data.default_root_state.clone()
                hand_state[:, 0:3] = safe_pos
                hand_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                hand_state[:, 7:] = 0.0
                self.robot.write_root_state_to_sim(hand_state)
                
                all_joint_pos = self.robot.data.default_joint_pos.clone()
                joint_vel = torch.zeros_like(all_joint_pos)
                self.robot.write_joint_state_to_sim(all_joint_pos, joint_vel, None)
                
                # ========== 2. RANDOMIZE NUMBER OF CLUTTER ==========
                total_available = len(self.objects) - 1
                
                if self.cfg.use_clutter_based_spawn and self.cfg.target_complexity:
                    from .benchmark_env_cfg import CLUTTER_CONFIGS
                    config = CLUTTER_CONFIGS[self.cfg.target_complexity]
                    
                    neighbor_range = config['num_neighbors']
                    if isinstance(neighbor_range, tuple):
                        min_neighbors, max_neighbors = neighbor_range
                    else:
                        min_neighbors = max_neighbors = neighbor_range
                    
                    num_clutter_objects = np.random.randint(min_neighbors, min(max_neighbors + 1, total_available + 1))
                    print(f"  Target clutter: {num_clutter_objects} objects (range: {min_neighbors}-{max_neighbors})")
                else:
                    num_clutter_objects = np.random.randint(1, total_available + 1)
                    print(f"  Random clutter: {num_clutter_objects} objects (max: {total_available})")
                
                # ========== 3. RANDOMLY SELECT WHICH CLUTTER OBJECTS ==========
                all_clutter_indices = list(range(1, len(self.objects)))
                np.random.shuffle(all_clutter_indices)
                
                active_clutter_indices = all_clutter_indices[:num_clutter_objects]
                inactive_clutter_indices = all_clutter_indices[num_clutter_objects:]
                
                print(f"  Active clutter: {[self._object_infos[i].object_id for i in active_clutter_indices[:3]]}{'...' if len(active_clutter_indices) > 3 else ''}")
                
                # ========== 4. TARGET AT CENTER (SAME position) ==========
                env_origins = self.scene.env_origins
                
                target_pos_world = env_origins.clone()
                target_pos_world[:, 0] += 0.0  # Center
                target_pos_world[:, 1] += 0.0  # Center
                target_pos_world[:, 2] += self.cfg.table_height + 0.05
                
                # Randomize orientation
                if self.cfg.randomize_object_orientation:
                    random_roll = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                    random_pitch = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                    random_yaw = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                    
                    random_euler = torch.stack([random_roll, random_pitch, random_yaw], dim=-1)
                    target_quat = euler_to_quaternion(random_euler)
                    target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
                else:
                    target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
                
                target_state = target_object.data.default_root_state.clone()
                target_state[:, 0:3] = target_pos_world
                target_state[:, 3:7] = target_quat
                target_state[:, 7:] = 0.0
                target_object.write_root_state_to_sim(target_state)
                
                print(f"  Target at CENTER (0.0, 0.0) - SAME")
                
                # ========== 5. POSITION CLUTTER (RANDOMIZED) ==========
                if num_clutter_objects > 0:
                    if self.cfg.use_clutter_based_spawn and self.cfg.target_complexity:
                        from .benchmark_env_cfg import CLUTTER_CONFIGS
                        config = CLUTTER_CONFIGS[self.cfg.target_complexity]
                        positions = self._generate_neighbor_positions(
                            center_pos=(0.0, 0.0),
                            num_neighbors=num_clutter_objects,
                            min_clearance=config['min_clearance'],
                            max_clearance=config['max_clearance']
                        )
                    else:
                        positions = self._generate_random_clutter_positions(num_clutter_objects)
                    
                    for i, clutter_idx in enumerate(active_clutter_indices):
                        clutter_obj = self.objects[clutter_idx]
                        pos_rel = positions[i]
                        
                        clutter_pos_world = env_origins.clone()
                        clutter_pos_world[:, 0] += pos_rel[0]
                        clutter_pos_world[:, 1] += pos_rel[1]
                        clutter_pos_world[:, 2] += self.cfg.table_height + 0.05
                        
                        if self.cfg.randomize_object_orientation:
                            random_roll = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                            random_pitch = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                            random_yaw = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                            
                            random_euler = torch.stack([random_roll, random_pitch, random_yaw], dim=-1)
                            clutter_quat = euler_to_quaternion(random_euler)
                            clutter_quat = clutter_quat / torch.norm(clutter_quat, dim=-1, keepdim=True)
                        else:
                            clutter_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
                        
                        clutter_state = clutter_obj.data.default_root_state.clone()
                        clutter_state[:, 0:3] = clutter_pos_world
                        clutter_state[:, 3:7] = clutter_quat
                        clutter_state[:, 7:] = 0.0
                        clutter_obj.write_root_state_to_sim(clutter_state)
                
                # ========== 6. HIDE INACTIVE CLUTTER ==========
                far_away = env_origins.clone()
                far_away[:, 0] = 50.0
                far_away[:, 1] = 50.0
                far_away[:, 2] = -20.0
                
                for clutter_idx in inactive_clutter_indices:
                    clutter_obj = self.objects[clutter_idx]
                    
                    clutter_state = clutter_obj.data.default_root_state.clone()
                    clutter_state[:, 0:3] = far_away
                    clutter_state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
                    clutter_state[:, 7:] = 0.0
                    clutter_obj.write_root_state_to_sim(clutter_state)
                
                print(f"  Positioned: {num_clutter_objects} active, {len(inactive_clutter_indices)} hidden")
                
                # ========== 7. SETTLE PHYSICS ==========
                for _ in range(self.cfg.spawn_settling_steps):
                    self.sim.step(render=False)
                
                for obj in self.objects:
                    obj.update(dt=self.cfg.sim.dt)
                
                for _ in range(30):
                    self.sim.step(render=False)
                
                for obj in self.objects:
                    obj.update(dt=self.cfg.sim.dt)
                
                # ========== 8. VALIDATE SPAWN ==========
                if not self._validate_spawn_configuration():
                    print(f"  ✗ Invalid spawn, retrying...")
                    continue
                
                print(f"  ✓ Valid spawn")
                
                # ========== 9. CAPTURE FRESH ANNOTATED IMAGE FOR THIS TRIAL ==========
                print(f"  Capturing FRESH annotated image for trial {self._current_trial + 1}...")
                
                # Render and capture
                for i in range(20):
                    self.sim.step(render=True)
                    if i % 5 == 0:
                        self.classifier_camera.update(dt=self.cfg.sim.dt)
                
                self.classifier_camera.update(dt=self.cfg.sim.dt)
                
                # Get camera data
                camera_data = self.classifier_camera.data
                cam_pos = camera_data.pos_w[0].clone()
                cam_quat = camera_data.quat_w_ros[0].clone()
                
                # Get RGB image
                if "rgb" not in self.classifier_camera.data.output:
                    print(f"  ✗ No RGB data")
                    continue
                
                rgb_data = self.classifier_camera.data.output["rgb"][0]
                rgb_np = rgb_data.cpu().numpy()
                
                if rgb_np.dtype != np.uint8:
                    base_image = (rgb_np * 255).astype(np.uint8)
                else:
                    base_image = rgb_np
                
                # ========== EXTRACT SPATIAL FEATURES FOR ALL ACTIVE OBJECTS ==========
                width = self.cfg.classifier_camera.width
                height = self.cfg.classifier_camera.height
                
                all_bbox_features = []
                all_spatial_features = []
                
                # Process target + active clutter only
                active_obj_indices = [0] + active_clutter_indices
                
                for obj_idx in active_obj_indices:
                    obj_info_local = self._object_infos[obj_idx]
                    
                    mesh_points_world = extract_policy_pointcloud_for_object(self, obj_idx, obj_info_local)
                    
                    if mesh_points_world is not None and len(mesh_points_world) > 0:
                        projected_2d = project_points_to_image_cached(
                            mesh_points_world, cam_pos, cam_quat, self.cfg.classifier_camera
                        )
                        
                        if projected_2d is not None:
                            xs, ys = projected_2d
                            
                            padding = 2
                            x_min = max(0, int(xs.min()) - padding)
                            x_max = min(width - 1, int(xs.max()) + padding)
                            y_min = max(0, int(ys.min()) - padding)
                            y_max = min(height - 1, int(ys.max()) + padding)
                            
                            bbox_features = {
                                'bbox_x_min': x_min / width,
                                'bbox_x_max': x_max / width,
                                'bbox_y_min': y_min / height,
                                'bbox_y_max': y_max / height,
                                'bbox_center_x': ((x_min + x_max) / 2) / width,
                                'bbox_center_y': ((y_min + y_max) / 2) / height,
                                'bbox_width': (x_max - x_min) / width,
                                'bbox_height': (y_max - y_min) / height,
                                'bbox_area': ((x_max - x_min) * (y_max - y_min)) / (width * height),
                            }
                            
                            position_features = estimate_3d_position_from_bbox(bbox_features)
                            
                            spatial_features = np.array([
                                position_features['obj_x'],
                                position_features['obj_y'],
                                position_features['distance_to_center'],
                                bbox_features['bbox_area'],
                                bbox_features['bbox_width'],
                                bbox_features['bbox_height'],
                            ], dtype=np.float32)
                            
                            all_bbox_features.append(bbox_features)
                            all_spatial_features.append(spatial_features)
                        else:
                            all_bbox_features.append(None)
                            all_spatial_features.append(None)
                    else:
                        all_bbox_features.append(None)
                        all_spatial_features.append(None)
                
                # ========== COUNT NEIGHBORS FOR TARGET ==========
                RADIUS = 0.06
                target_neighbor_count = 0
                
                if all_spatial_features[0] is not None:
                    target_x = all_spatial_features[0][0]
                    target_y = all_spatial_features[0][1]
                    
                    for other_idx in range(1, len(all_spatial_features)):
                        if all_spatial_features[other_idx] is None:
                            continue
                        
                        other_x = all_spatial_features[other_idx][0]
                        other_y = all_spatial_features[other_idx][1]
                        
                        distance = np.sqrt((target_x - other_x)**2 + (target_y - other_y)**2)
                        
                        if distance < RADIUS:
                            target_neighbor_count += 1
                
                print(f"  Target has {target_neighbor_count} neighbors (radius={RADIUS}m)")
                
                # ========== ANNOTATE IMAGE WITH CONSISTENT STYLE ==========
                annotated_image = base_image.copy()
                
                # Draw target (GREEN box)
                if all_bbox_features[0] is not None:
                    bbox = all_bbox_features[0]
                    x_min = int(bbox['bbox_x_min'] * width)
                    x_max = int(bbox['bbox_x_max'] * width)
                    y_min = int(bbox['bbox_y_min'] * height)
                    y_max = int(bbox['bbox_y_max'] * height)
                    
                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), 
                                (50, 255, 50), 3)  # Green, thick
                
                # Draw clutter objects (YELLOW for neighbors, ORANGE for non-neighbors)
                neighbors_drawn = 0
                non_neighbors_drawn = 0
                
                for i in range(1, len(all_spatial_features)):
                    if all_spatial_features[i] is None:
                        continue
                    
                    # Check if neighbor
                    obj_x = all_spatial_features[i][0]
                    obj_y = all_spatial_features[i][1]
                    target_x = all_spatial_features[0][0]
                    target_y = all_spatial_features[0][1]
                    
                    distance = np.sqrt((obj_x - target_x)**2 + (obj_y - target_y)**2)
                    
                    bbox = all_bbox_features[i]
                    if bbox is not None:
                        x_min = int(bbox['bbox_x_min'] * width)
                        x_max = int(bbox['bbox_x_max'] * width)
                        y_min = int(bbox['bbox_y_min'] * height)
                        y_max = int(bbox['bbox_y_max'] * height)
                        
                        if distance < RADIUS:
                            # NEIGHBOR - Yellow
                            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), 
                                        (0, 255, 255), 2)
                            neighbors_drawn += 1
                        else:
                            # NON-NEIGHBOR - Orange
                            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), 
                                        (0, 165, 255), 2)
                            non_neighbors_drawn += 1
                
                print(f"  Drew {neighbors_drawn} neighbor boxes (yellow)")
                print(f"  Drew {non_neighbors_drawn} non-neighbor boxes (orange)")
                
                # ========== ADD LABELS ==========
                trial_text = f"Trial {self._current_trial + 1}/{self.cfg.num_trials}"
                mode_text = f"Clutter: {self.cfg.target_complexity if self.cfg.target_complexity else 'Random'}"
                object_text = f"Target: {obj_info.object_id}"
                neighbor_text = f"Neighbors: {target_neighbor_count}"
                other_text = f"Non-neighbors: {non_neighbors_drawn}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                
                # White background for labels
                y_offset = 30
                for text in [trial_text, mode_text, object_text, neighbor_text, other_text]:
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, font, font_scale, font_thickness
                    )
                    
                    cv2.rectangle(annotated_image, 
                                (10, y_offset - text_height - 5), 
                                (20 + text_width, y_offset + 5),
                                (255, 255, 255), -1)
                    
                    cv2.putText(annotated_image, text, (15, y_offset),
                            font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
                    
                    y_offset += text_height + 15
                
                # Add legend
                legend_x = width - 200
                legend_y = height - 120
                
                cv2.rectangle(annotated_image, 
                            (legend_x - 10, legend_y - 10),
                            (width - 10, height - 10),
                            (255, 255, 255), -1)
                
                legend_items = [
                    ("Target", (50, 255, 50)),
                    ("Neighbor", (0, 255, 255)),
                    ("Non-neighbor", (0, 165, 255))
                ]
                
                y_pos = legend_y + 10
                for label_text, color in legend_items:
                    cv2.rectangle(annotated_image,
                                (legend_x, y_pos - 10),
                                (legend_x + 20, y_pos + 5),
                                color, -1)
                    
                    cv2.putText(annotated_image, label_text,
                            (legend_x + 30, y_pos),
                            font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    y_pos += 25
                
                # ========== SAVE ANNOTATED IMAGE ==========
                image_name = f"trial_{self._current_trial + 1:02d}_clutter_{self.cfg.target_complexity}.png"
                image_path = Path(self.cfg.classifier_image_dir) / image_name
                if self._current_trial % 10:
                    cv2.imwrite(str(image_path), annotated_image)
                
                    print(f"  ✓ Annotated image saved: {image_name}")
                
                # ========== COMPLEXITY VERIFICATION (RUNS EVERY TRIAL NOW!) ==========
                # CRITICAL FIX: Check for self._classifier_model instead of undefined variable
                if (hasattr(self, '_classifier_model') and self._classifier_model and 
                    self.cfg.verify_target_complexity and 
                    hasattr(self, '_predict_complexity_fn') and self._predict_complexity_fn):
                    
                    print(f"\n[VERIFY] Running classifier for trial {self._current_trial + 1}...")
                    
                    # Save temp image with only target box for classifier
                    temp_img = base_image.copy()
                    if all_bbox_features[0] is not None:
                        bbox = all_bbox_features[0]
                        x_min = int(bbox['bbox_x_min'] * width)
                        x_max = int(bbox['bbox_x_max'] * width)
                        y_min = int(bbox['bbox_y_min'] * height)
                        y_max = int(bbox['bbox_y_max'] * height)
                        cv2.rectangle(temp_img, (x_min, y_min), (x_max, y_max), (50, 255, 50), 2)
                    
                    temp_path = Path(self.cfg.classifier_image_dir) / f"temp_verify_trial_{self._current_trial + 1}.png"
                    cv2.imwrite(str(temp_path), temp_img)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # CRITICAL: Use self._predict_complexity_fn with verbose=True
                    predicted_complexity = self._predict_complexity_fn(
                        self._classifier_model, str(temp_path), device, verbose=True
                    )
                    
                    from .benchmark_env_cfg import CLUTTER_CONFIGS
                    complexity_map = {'C0_easy': 0, 'C1_medium': 1, 'C2_hard': 2}
                    desired_level = complexity_map[self.cfg.target_complexity]
                    
                    # Apply correction based on neighbor count
                    corrected_complexity = _apply_complexity_correction(
                        predicted_complexity, target_neighbor_count,
                        obj_name=f"Trial {self._current_trial + 1}"
                    )
                    
                    print(f"\n[VERIFY] Classification Results:")
                    print(f"  Trial: {self._current_trial + 1}")
                    print(f"  Object: {obj_info.object_id}")
                    print(f"  Raw Prediction: C{predicted_complexity}")
                    print(f"  Neighbor Count: {target_neighbor_count}")
                    print(f"  Corrected: C{corrected_complexity}")
                    print(f"  Desired: C{desired_level}")
                    
                    temp_path.unlink()
                    
                    if corrected_complexity == desired_level:
                        print(f"  ✓ VERIFIED for trial {self._current_trial + 1}!")
                        verified = True
                        break
                    else:
                        print(f"  ✗ Mismatch (C{corrected_complexity} != C{desired_level}), retrying...")
                        continue
                else:
                    # No classifier available or verification disabled
                    print(f"  [INFO] Classifier verification disabled or not available")
                    verified = True
                    break
            
            if not verified:
                print(f"[TRIAL] [WARN] Proceeding without perfect verification")
            
            print(f"[TRIAL] ✓ Scene ready with fresh annotated image")
        
        # ========== CAPTURE SETTLED POSITION ==========
        target_object.update(dt=self.cfg.sim.dt)
        target_pos_world = target_object.data.root_pos_w[0].cpu()
        
        actual_x = float(target_pos_world[0].item())
        actual_y = float(target_pos_world[1].item())
        actual_z = float(target_pos_world[2].item())
        
        self.target_initial_pos = {0: (actual_x, actual_y)}
        self.target_last_table_pos = {0: (actual_x, actual_y)}
        self.target_status[0] = 'attempting'
        
        self._trial_initial_positions.append(self.target_initial_pos.copy())
        
        print(f"[TRIAL] Target settled: ({actual_x:.4f}, {actual_y:.4f}, z={actual_z:.4f})")
        print(f"[TRIAL] ✓ Initial position captured")
        print(f"{'='*80}\n")

    def _extract_object_mesh_points(self):
        """Extract collision mesh vertices directly from PhysX (most MuJoCo-like).
        
        This queries the actual collision mesh being used by the physics engine,
        similar to how MuJoCo accesses model.mesh_vert[].
        """
        try:
            
            stage = omni.usd.get_context().get_stage()
            object_prim_path = f"{self.scene.env_prim_paths[0]}/Object"
            
            if self.cfg.verbose_pointcloud_extraction:
                print(f"[INFO] Querying PhysX collision mesh from: {object_prim_path}")
            
            # Method 1: Try to get mesh from collision shape
            mesh_vertices = self._get_physx_collision_mesh(object_prim_path)
            
            if mesh_vertices is None or len(mesh_vertices) == 0:
                print("[WARN] Could not get collision mesh, trying convex hull approximation...")
                mesh_vertices = self._get_convex_hull_mesh(object_prim_path)
            
            if mesh_vertices is None or len(mesh_vertices) == 0:
                print("[WARN] Could not extract mesh, using default point cloud")
                mesh_points = self._generate_default_point_cloud()
            else:
                print(f"[INFO] Extracted {len(mesh_vertices)} vertices from collision mesh")
                
                # Auto-detect scale
                max_coord = np.abs(mesh_vertices).max()
                print(f"[INFO] Max coordinate: {max_coord:.6f}")
                
                if max_coord > 10.0:
                    mesh_vertices *= 0.001
                    print("[INFO] Applied mm->m conversion")
                elif max_coord > 1.0:
                    mesh_vertices *= 0.01
                    print("[INFO] Applied cm->m conversion")
                
                print(f"[INFO] Mesh bounds (local frame):")
                print(f"  X: [{mesh_vertices[:, 0].min():.4f}, {mesh_vertices[:, 0].max():.4f}]")
                print(f"  Y: [{mesh_vertices[:, 1].min():.4f}, {mesh_vertices[:, 1].max():.4f}]")
                print(f"  Z: [{mesh_vertices[:, 2].min():.4f}, {mesh_vertices[:, 2].max():.4f}]")
                
                # MuJoCo-style subsampling
                mesh_points = self._subsample_vertices(mesh_vertices, self.cfg.num_object_pc_points)
            
            # Store in local frame (like MuJoCo's mesh_vert)
            self._object_mesh_points_local = torch.tensor(
                mesh_points,
                device=self.device,
                dtype=torch.float32
            )
            
            print(f"[INFO] Cached {len(mesh_points)} vertices in LOCAL object frame")
            
        except Exception as e:
            print(f"[ERROR] Failed to extract mesh: {e}")
            import traceback
            traceback.print_exc()
            
            mesh_points = self._generate_default_point_cloud()
            self._object_mesh_points_local = torch.tensor(
                mesh_points,
                device=self.device,
                dtype=torch.float32
            )

    def _extract_object_mesh_for_current(self, obj_idx: int):
        """Extract mesh directly from the spawned object in the simulation.
        
        This is MORE RELIABLE than extracting from the source USD file because:
        1. It accounts for any scale/transform changes during spawn
        2. It matches exactly what the physics engine sees
        3. It's consistent with the object's actual size in simulation
        
        This is the CORRECT approach for multi-object sequential picking.
        """
        obj_info = self._object_infos[obj_idx]
        
        print(f"\n{'='*80}")
        print(f"[MESH EXTRACT] Extracting mesh from SPAWNED object: {obj_info.object_id}")
        print(f"{'='*80}")
        
        try:
            # Get the spawned object's prim path in the simulation
            stage = omni.usd.get_context().get_stage()
            env_path = self.scene.env_prim_paths[0]  # First environment
            object_prim_path = f"{env_path}/Object_{obj_idx}"
            
            print(f"[MESH EXTRACT] Prim path: {object_prim_path}")
            
            # Get the prim
            object_prim = stage.GetPrimAtPath(object_prim_path)
            
            if not object_prim.IsValid():
                raise RuntimeError(f"Object prim not valid: {object_prim_path}")
            
            # Recursively find all mesh prims under this object
            all_vertices = []
            mesh_count = 0
            
            def collect_mesh_vertices(prim, parent_xform=Gf.Matrix4d(1.0)):
                """Recursively collect vertices from all mesh children."""
                nonlocal mesh_count
                
                # Get this prim's local transform
                if prim.IsA(UsdGeom.Xformable):
                    xformable = UsdGeom.Xformable(prim)
                    local_xform = xformable.GetLocalTransformation()
                    current_xform = parent_xform * local_xform
                else:
                    current_xform = parent_xform
                
                # If this is a mesh, extract vertices
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points_attr = mesh.GetPointsAttr()
                    
                    if points_attr and points_attr.Get():
                        points = points_attr.Get()
                        # Convert to numpy
                        vertices = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points])
                        
                        # Apply accumulated transform (to get vertices in object's local frame)
                        if current_xform != Gf.Matrix4d(1.0):
                            # Transform vertices
                            vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
                            xform_np = np.array([
                                [current_xform[i][j] for j in range(4)] for i in range(4)
                            ])
                            vertices_transformed = (xform_np @ vertices_homogeneous.T).T
                            vertices = vertices_transformed[:, :3]
                        
                        all_vertices.append(vertices)
                        mesh_count += 1
                        print(f"[MESH EXTRACT] Found mesh {mesh_count}: {len(vertices)} vertices at {prim.GetPath()}")
                
                # Recurse to children
                for child in prim.GetChildren():
                    collect_mesh_vertices(child, current_xform)
            
            # Start collection from object root
            collect_mesh_vertices(object_prim)
            
            if len(all_vertices) == 0:
                print(f"[ERROR] No meshes found in spawned object!")
                return self._generate_default_point_cloud()
            
            # Combine all meshes
            mesh_vertices = np.vstack(all_vertices)
            print(f"[MESH EXTRACT] Combined {mesh_count} meshes = {len(mesh_vertices)} total vertices")
            
            # Check scale
            max_coord = np.abs(mesh_vertices).max()
            print(f"[MESH EXTRACT] Max coordinate: {max_coord:.6f}")
            
            # Auto-scale if needed
            if max_coord > 10.0:
                mesh_vertices *= 0.001
                print(f"[MESH EXTRACT] Applied mm->m conversion (÷1000)")
            elif max_coord > 1.0:
                mesh_vertices *= 0.01
                print(f"[MESH EXTRACT] Applied cm->m conversion (÷100)")
            
            # Print bounds
            print(f"[MESH EXTRACT] Mesh bounds (local frame):")
            print(f"  X: [{mesh_vertices[:, 0].min():.4f}, {mesh_vertices[:, 0].max():.4f}]")
            print(f"  Y: [{mesh_vertices[:, 1].min():.4f}, {mesh_vertices[:, 1].max():.4f}]")
            print(f"  Z: [{mesh_vertices[:, 2].min():.4f}, {mesh_vertices[:, 2].max():.4f}]")
            
            # Compute bounding box size
            bbox_size = np.array([
                mesh_vertices[:, 0].max() - mesh_vertices[:, 0].min(),
                mesh_vertices[:, 1].max() - mesh_vertices[:, 1].min(),
                mesh_vertices[:, 2].max() - mesh_vertices[:, 2].min()
            ])
            print(f"[MESH EXTRACT] Bounding box size: ({bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f})")
            
            # Subsample
            mesh_points = self._subsample_vertices(mesh_vertices, self.cfg.num_object_pc_points)
            
            # Store in LOCAL frame
            self._object_mesh_points_local = torch.tensor(
                mesh_points,
                device=self.device,
                dtype=torch.float32
            )
            
            print(f"[MESH EXTRACT] ✓ Cached {len(mesh_points)} points in LOCAL frame")
            print(f"{'='*80}\n")
            
            return mesh_points
            
        except Exception as e:
            print(f"[ERROR] Failed to extract mesh from spawned object: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to default point cloud
            mesh_points = self._generate_default_point_cloud()
            self._object_mesh_points_local = torch.tensor(
                mesh_points,
                device=self.device,
                dtype=torch.float32
            )
            return mesh_points

    def _check_pick_success(self) -> bool:
        """Check if current pick was successful."""
        if self._current_object_idx >= len(self._picking_order):
            return False
        
        obj_idx = self._picking_order[self._current_object_idx]
        current_object = self.objects[obj_idx]
        object_pos_world = current_object.data.root_pos_w[0]
        object_init_pos_world = self.object_init_pos[0]
        
        height_change = object_pos_world[2] - object_init_pos_world[2]
        success = height_change >= (self.cfg.max_lift_height * 0.6)
        
        return success.item()

    def _get_physx_collision_mesh(self, object_prim_path: str) -> np.ndarray:
        """Query the actual PhysX collision mesh."""
        stage = omni.usd.get_context().get_stage()
        
        def recursive_find_mesh(prim, parent_xform=None):
            """Recursively find mesh and accumulate transforms."""
            # Get this prim's transform
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)
                local_xform = xformable.GetLocalTransformation()
                
                if parent_xform is not None:
                    # Accumulate transforms
                    current_xform = parent_xform * local_xform
                else:
                    current_xform = local_xform
            else:
                current_xform = parent_xform
            
            # Check if this is a mesh
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                
                if points_attr and points_attr.Get():
                    points = points_attr.Get()
                    vertices = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points])
                    
                    # Apply accumulated transformation
                    if current_xform is not None:
                        vertices = self._transform_vertices(vertices, current_xform)
                    
                    return vertices
            
            # Recurse to children
            for child in prim.GetChildren():
                result = recursive_find_mesh(child, current_xform)
                if result is not None:
                    return result
            
            return None
        
        prim = stage.GetPrimAtPath(object_prim_path)
        return recursive_find_mesh(prim)


    def _get_convex_hull_mesh(self, object_prim_path: str) -> np.ndarray:
        """Get convex hull of the object mesh (approximation)."""
        try:
      
            # First get raw vertices
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(object_prim_path)
            
            def find_mesh(prim):
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points_attr = mesh.GetPointsAttr()
                    if points_attr and points_attr.Get():
                        points = points_attr.Get()
                        return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points])
                
                for child in prim.GetChildren():
                    result = find_mesh(child)
                    if result is not None:
                        return result
                return None
            
            vertices = find_mesh(prim)
            
            if vertices is not None and len(vertices) > 4:
                # Compute convex hull
                hull = ConvexHull(vertices)
                hull_vertices = vertices[hull.vertices]
                print(f"[INFO] Computed convex hull: {len(hull_vertices)} vertices")
                return hull_vertices
            
            return None
            
        except Exception as e:
            print(f"[WARN] Convex hull computation failed: {e}")
            return None


    def _transform_vertices(self, vertices: np.ndarray, transform_matrix) -> np.ndarray:
        """Apply 4x4 transformation matrix to vertices."""
        # Convert vertices to homogeneous coordinates
        ones = np.ones((vertices.shape[0], 1))
        verts_homogeneous = np.hstack([vertices, ones])
        
        # Apply transformation
        transform_np = np.array([
            [transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], transform_matrix[0][3]],
            [transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], transform_matrix[1][3]],
            [transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], transform_matrix[2][3]],
            [transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2], transform_matrix[3][3]],
        ])
        
        transformed = verts_homogeneous @ transform_np.T
        
        # Convert back to 3D
        return transformed[:, :3]


    def _subsample_vertices(self, vertices: np.ndarray, target_count: int) -> np.ndarray:
        """MuJoCo-style vertex subsampling."""
        if len(vertices) > target_count:
            # Random subsampling
            indices = np.random.choice(len(vertices), target_count, replace=False)
            return vertices[indices]
        elif len(vertices) < target_count:
            # Upsample with noise
            repeats = (target_count // len(vertices)) + 1
            upsampled = np.tile(vertices, (repeats, 1))[:target_count]
            noise = np.random.normal(0, 0.0005, upsampled.shape)
            return upsampled + noise
        else:
            return vertices


    def _generate_default_point_cloud(self) -> np.ndarray:
        """Generate default spherical point cloud."""
        n = self.cfg.num_object_pc_points
        radius = 0.03
        
        points = []
        phi = np.pi * (3. - np.sqrt(5.))
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius_at_y = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append([x * radius, y * radius, z * radius])
        
        return np.array(points)

    def _sample_object_point_cloud(self) -> torch.Tensor:
        """Sample points on current target object surface with verification."""
        if self._object_mesh_points_local is None:
            print(f"[ERROR] No mesh points loaded!")
            return torch.zeros((self.num_envs, self.cfg.num_object_pc_points, 3), device=self.device)
        
        # Safety check
        if self._current_object_idx >= len(self._picking_order):
            print(f"[ERROR] Current object index out of range!")
            return torch.zeros((self.num_envs, self.cfg.num_object_pc_points, 3), device=self.device)
        
        # Get current target object
        obj_idx = self._picking_order[self._current_object_idx]
        current_object = self.objects[obj_idx]
        object_pos = current_object.data.root_pos_w
        object_quat = current_object.data.root_quat_w
        
        # Transform local points to world
        local_points = self._object_mesh_points_local.unsqueeze(0).expand(self.num_envs, -1, -1)
        world_points = quat_apply_batch(object_quat, local_points)
        world_points = world_points + object_pos.unsqueeze(1)
        
        # Debug: Verify point cloud is near object
        if self._debug_step_counter % 100 == 0 and self.cfg.enable_debug_obs:
            pc_center = world_points[0].mean(dim=0)
            obj_pos_0 = object_pos[0]
            distance = torch.norm(pc_center - obj_pos_0).item()
            print(f"[DEBUG] Point cloud center distance from object: {distance:.4f}m")
            if distance > 0.1:
                print(f"[WARN] ⚠ Point cloud is far from object!")
        
        return world_points

    def _setup_point_cloud_markers(self):
        """Setup visualization markers for point cloud."""
        if self._pc_markers_initialized:
            return
        
        stage = omni.usd.get_context().get_stage()
        
        for env_idx in range(self._num_vis_envs):
            env_path = self.scene.env_prim_paths[env_idx]
            markers_path = f"{env_path}/PointCloudMarkers"
            
            if stage.GetPrimAtPath(markers_path).IsValid():
                continue
            
            xform = UsdGeom.Xform.Define(stage, markers_path)
            
            for i in range(self.cfg.num_object_pc_points):
                marker_cfg = sim_utils.SphereCfg(
                    radius=0.002,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=False,
                        kinematic_enabled=True,
                    ),
                )
                marker_cfg.func(f"{markers_path}/Point_{i}", marker_cfg, translation=(0, 0, 0))
        
        self._pc_markers_initialized = True
        print(f"[INFO] Created point cloud markers for {self._num_vis_envs} environments")

    def _setup_site_markers(self):
        """Setup visualization markers for contact sites."""
        if self._site_markers_initialized:
            return
        
        stage = omni.usd.get_context().get_stage()
        
        for env_idx in range(self._num_vis_envs):
            env_path = self.scene.env_prim_paths[env_idx]
            
            # Finger sites - BLUE
            finger_markers_path = f"{env_path}/FingerSiteMarkers"
            if not stage.GetPrimAtPath(finger_markers_path).IsValid():
                xform = UsdGeom.Xform.Define(stage, finger_markers_path)
                
                for i in range(10):
                    marker_cfg = sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=False,
                            kinematic_enabled=True,
                        ),
                    )
                    marker_cfg.func(f"{finger_markers_path}/FingerSite_{i}", marker_cfg, translation=(0, 0, 0))
            
            # Palm sites - GREEN
            palm_markers_path = f"{env_path}/PalmSiteMarkers"
            if not stage.GetPrimAtPath(palm_markers_path).IsValid():
                xform = UsdGeom.Xform.Define(stage, palm_markers_path)
                
                for i in range(7):
                    marker_cfg = sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=False,
                            kinematic_enabled=True,
                        ),
                    )
                    marker_cfg.func(f"{palm_markers_path}/PalmSite_{i}", marker_cfg, translation=(0, 0, 0))
        
        self._site_markers_initialized = True
        print(f"[INFO] Created site markers for {self._num_vis_envs} environments")

    def _setup_object_root_markers(self):
        """Setup visualization markers for object root positions."""
        if self._object_root_markers_initialized:
            return
        
        stage = omni.usd.get_context().get_stage()
        
        print(f"[INFO] Setting up object root markers for {len(self.objects)} objects...")
        
        for env_idx in range(self._num_vis_envs):
            env_path = self.scene.env_prim_paths[env_idx]
            markers_path = f"{env_path}/ObjectRootMarkers"
            
            if stage.GetPrimAtPath(markers_path).IsValid():
                continue
            
            # Create parent xform
            xform = UsdGeom.Xform.Define(stage, markers_path)
            
            # Create a marker for each object
            for obj_idx in range(len(self.objects)):
                marker_cfg = sim_utils.SphereCfg(
                    radius=0.01,  # 1cm sphere - larger than point cloud points
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0),  # Yellow
                        emissive_color=(0.3, 0.3, 0.0),  # Slight glow
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=False,
                        kinematic_enabled=True,
                    ),
                )
                marker_cfg.func(
                    f"{markers_path}/ObjectRoot_{obj_idx}", 
                    marker_cfg, 
                    translation=(0, 0, 0)
                )
        
        self._object_root_markers_initialized = True
        print(f"[INFO] Created object root markers for {len(self.objects)} objects")

    def _update_point_cloud_visualization(self):
        """Update point cloud markers with WORLD coordinates."""
        if not self.cfg.visualize_point_cloud or self._object_mesh_points_local is None:
            return
        
        if not self._pc_markers_initialized:
            return
        
        # Get point cloud in WORLD coordinates
        point_cloud_world = self._sample_object_point_cloud()
        
        stage = omni.usd.get_context().get_stage()
        
        # Update only visualized environments
        for env_idx in range(self._num_vis_envs):
            env_points = point_cloud_world[env_idx].cpu().numpy()
            env_path = self.scene.env_prim_paths[env_idx]
            markers_path = f"{env_path}/PointCloudMarkers"
            
            for i, point in enumerate(env_points):
                marker_path = f"{markers_path}/Point_{i}"
                prim = stage.GetPrimAtPath(marker_path)
                if prim.IsValid():
                    xformable = UsdGeom.Xformable(prim)
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp()
                    # Use WORLD coordinates directly
                    translate_op.Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))

    def _update_site_visualization(self):
        """Update site markers with WORLD coordinates."""
        if not self.cfg.visualize_point_cloud or not self._site_markers_initialized:
            return
        
        # Get site positions in WORLD coordinates
        finger_site_pos, palm_site_pos = self._get_site_positions()
        
        stage = omni.usd.get_context().get_stage()
        
        for env_idx in range(self._num_vis_envs):
            env_path = self.scene.env_prim_paths[env_idx]
            
            # Update finger sites
            finger_markers_path = f"{env_path}/FingerSiteMarkers"
            for i in range(10):
                marker_path = f"{finger_markers_path}/FingerSite_{i}"
                prim = stage.GetPrimAtPath(marker_path)
                if prim.IsValid():
                    pos = finger_site_pos[env_idx, i].cpu().numpy()
                    xformable = UsdGeom.Xformable(prim)
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            
            # Update palm sites
            palm_markers_path = f"{env_path}/PalmSiteMarkers"
            for i in range(7):
                marker_path = f"{palm_markers_path}/PalmSite_{i}"
                prim = stage.GetPrimAtPath(marker_path)
                if prim.IsValid():
                    pos = palm_site_pos[env_idx, i].cpu().numpy()
                    xformable = UsdGeom.Xformable(prim)
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp()
                    translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def _update_object_root_visualization(self):
        """Update object root position markers with WORLD coordinates."""
        if not self.cfg.visualize_point_cloud or not self._object_root_markers_initialized:
            return
        
        stage = omni.usd.get_context().get_stage()
        
        # Update only visualized environments
        for env_idx in range(self._num_vis_envs):
            env_path = self.scene.env_prim_paths[env_idx]
            markers_path = f"{env_path}/ObjectRootMarkers"
            
            # Update each object's root marker
            for obj_idx, obj in enumerate(self.objects):
                # Get object root position in WORLD coordinates
                obj_pos_world = obj.data.root_pos_w[env_idx].cpu().numpy()
                
                marker_path = f"{markers_path}/ObjectRoot_{obj_idx}"
                prim = stage.GetPrimAtPath(marker_path)
                
                if prim.IsValid():
                    xformable = UsdGeom.Xformable(prim)
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp()
                    # Use WORLD coordinates directly
                    translate_op.Set(Gf.Vec3d(
                        float(obj_pos_world[0]), 
                        float(obj_pos_world[1]), 
                        float(obj_pos_world[2])
                    ))

    def _verify_mesh_extraction(self, obj_idx: int):
        """Verify that extracted mesh actually belongs to the target object.
        
        This compares the bounding box of the extracted mesh with the object's
        actual bounding box in the simulation.
        """
        obj_info = self._object_infos[obj_idx]
        current_object = self.objects[obj_idx]
        
        print(f"\n{'='*80}")
        print(f"[MESH VERIFY] Checking mesh for: {obj_info.object_id}")
        print(f"{'='*80}")
        
        # 1. Get object's bounding box from simulation
        obj_pos_world = current_object.data.root_pos_w[0]
        obj_quat = current_object.data.root_quat_w[0:1]
        
        print(f"[MESH VERIFY] Object position (world): {obj_pos_world}")
        
        # 2. Get extracted mesh bounding box
        if self._object_mesh_points_local is None:
            print(f"[ERROR] No mesh points extracted!")
            return False
        
        mesh_local = self._object_mesh_points_local
        mesh_bbox_local = {
            'min': mesh_local.min(dim=0)[0],
            'max': mesh_local.max(dim=0)[0],
            'center': mesh_local.mean(dim=0),
            'size': (mesh_local.max(dim=0)[0] - mesh_local.min(dim=0)[0])
        }
        
        print(f"[MESH VERIFY] Mesh bounding box (local frame):")
        print(f"  Min: {mesh_bbox_local['min']}")
        print(f"  Max: {mesh_bbox_local['max']}")
        print(f"  Center: {mesh_bbox_local['center']}")
        print(f"  Size: {mesh_bbox_local['size']}")
        
        # 3. Transform mesh to world coordinates
        local_points = mesh_local.unsqueeze(0)
        world_points = quat_apply_batch(obj_quat, local_points)
        world_points = world_points + obj_pos_world.unsqueeze(0).unsqueeze(1)
        world_points = world_points[0]  # Remove batch dim
        
        mesh_bbox_world = {
            'min': world_points.min(dim=0)[0],
            'max': world_points.max(dim=0)[0],
            'center': world_points.mean(dim=0),
            'size': (world_points.max(dim=0)[0] - world_points.min(dim=0)[0])
        }
        
        print(f"\n[MESH VERIFY] Mesh bounding box (world frame):")
        print(f"  Min: {mesh_bbox_world['min']}")
        print(f"  Max: {mesh_bbox_world['max']}")
        print(f"  Center: {mesh_bbox_world['center']}")
        print(f"  Size: {mesh_bbox_world['size']}")
        
        # 4. Compare with object position
        center_offset = torch.norm(mesh_bbox_world['center'] - obj_pos_world).item()
        print(f"\n[MESH VERIFY] Center offset: {center_offset:.4f}m")
        
        # 5. Get actual object size from USD (if possible)
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
            
            stage = Usd.Stage.Open(obj_info.usd_path)
            if stage:
                # Get bounding box from USD
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
                root_prim = stage.GetDefaultPrim()
                if root_prim:
                    bbox = bbox_cache.ComputeWorldBound(root_prim)
                    bbox_range = bbox.ComputeAlignedRange()
                    usd_size = bbox_range.GetSize()
                    
                    print(f"\n[MESH VERIFY] USD bounding box size: ({usd_size[0]:.4f}, {usd_size[1]:.4f}, {usd_size[2]:.4f})")
                    print(f"[MESH VERIFY] Mesh size: {mesh_bbox_local['size']}")
                    
                    # Compare sizes
                    size_diff = torch.tensor([
                        abs(mesh_bbox_local['size'][0].item() - usd_size[0]),
                        abs(mesh_bbox_local['size'][1].item() - usd_size[1]),
                        abs(mesh_bbox_local['size'][2].item() - usd_size[2])
                    ])
                    
                    print(f"[MESH VERIFY] Size difference: {size_diff}")
                    
                    if size_diff.max().item() > 0.02:  # 2cm tolerance
                        print(f"[ERROR] ❌ Mesh size does NOT match USD file!")
                        print(f"[ERROR] This confirms the wrong mesh was extracted!")
                        return False
                    else:
                        print(f"[SUCCESS] ✓ Mesh size matches USD file")
        except Exception as e:
            print(f"[WARN] Could not load USD for comparison: {e}")
        
        # 6. Visual check
        if center_offset > 0.10:  # 10cm
            print(f"\n[ERROR] ❌ Point cloud center is FAR from object!")
            print(f"[ERROR] Expected offset < 0.10m, got {center_offset:.4f}m")
            return False
        elif center_offset > 0.05:  # 5cm
            print(f"\n[WARN] ⚠ Point cloud center offset is large: {center_offset:.4f}m")
            print(f"[WARN] Acceptable but may affect policy performance")
            return True
        else:
            print(f"\n[SUCCESS] ✓ Point cloud is well-aligned (offset: {center_offset:.4f}m)")
            return True
    
    def _verify_policy_object_match(self, obj_info):
        """Verify that the loaded policy was actually trained for this object."""
        
        print(f"\n{'='*80}")
        print(f"[POLICY VERIFY] Checking policy for: {obj_info.object_id}")
        print(f"{'='*80}")
        
        policy_path = Path(obj_info.policy_path)
        
        # 1. Check directory structure
        policy_dir = policy_path.parent
        dir_name = policy_dir.name
        
        print(f"[POLICY VERIFY] Policy directory: {policy_dir}")
        print(f"[POLICY VERIFY] Directory name: {dir_name}")
        print(f"[POLICY VERIFY] Object ID: {obj_info.object_id}")
        
        if dir_name != obj_info.object_id:
            print(f"[ERROR] ❌ Policy directory name does NOT match object ID!")
            print(f"[ERROR] Expected: {obj_info.object_id}")
            print(f"[ERROR] Got: {dir_name}")
            return False
        
        # 2. Check if USD file name matches
        usd_path = Path(obj_info.usd_path)
        usd_name = usd_path.stem  # Filename without extension
        
        print(f"[POLICY VERIFY] USD filename: {usd_name}")
        
        if usd_name != obj_info.object_id:
            print(f"[WARN] ⚠ USD filename does NOT match object ID!")
            print(f"[WARN] Expected: {obj_info.object_id}")
            print(f"[WARN] Got: {usd_name}")
        
        # 3. Check if policy file exists
        if not policy_path.exists():
            print(f"[ERROR] ❌ Policy file does not exist: {policy_path}")
            return False
        
        # 4. Try to extract metadata from policy (if available)
        try:
            from stable_baselines3 import PPO
            import zipfile
            
            # SB3 policies are zip files, check contents
            with zipfile.ZipFile(policy_path, 'r') as z:
                files = z.namelist()
                print(f"\n[POLICY VERIFY] Policy archive contains:")
                for f in files:
                    print(f"  - {f}")
                
                # Check for data file
                if 'data' in files:
                    data_content = z.read('data').decode('utf-8')
                    if obj_info.object_id in data_content:
                        print(f"[SUCCESS] ✓ Found object ID in policy metadata")
                    else:
                        print(f"[WARN] ⚠ Object ID not found in policy metadata")
        except Exception as e:
            print(f"[WARN] Could not inspect policy archive: {e}")
        
        print(f"\n[POLICY VERIFY] Policy appears to be correct for {obj_info.object_id}")
        return True
    
    def _verify_policy_output(self):
        """Fixed version that handles CUDA tensors properly."""
        
        if self._current_policy is None:
            print(f"[ERROR] No policy loaded!")
            return False
        
        print(f"\n{'='*80}")
        print(f"[POLICY OUTPUT] Testing policy predictions")
        print(f"{'='*80}")
        
        # Get current observation
        obs_dict = self._get_observations()
        obs = obs_dict["policy"]
        
        # Convert to numpy on CPU for SB3
        obs_np = obs.cpu().numpy()
        
        # Run policy multiple times
        actions_list = []
        for i in range(5):
            with torch.inference_mode():
                # SB3 expects numpy arrays
                action, _ = self._current_policy.predict(obs_np, deterministic=True)
            actions_list.append(action)
        
        actions_array = np.array(actions_list)
        
        print(f"[POLICY OUTPUT] Actions over 5 predictions:")
        print(f"  Shape: {actions_array.shape}")
        print(f"  Min: {actions_array.min(axis=0)}")
        print(f"  Max: {actions_array.max(axis=0)}")
        print(f"  Mean: {actions_array.mean(axis=0)}")
        print(f"  Std: {actions_array.std(axis=0)}")
        
        # Check if actions are constant
        if actions_array.std() < 0.001:
            print(f"[ERROR] ❌ Policy produces nearly constant actions!")
            return False
        
        # Check range
        if actions_array.min() < -1.1 or actions_array.max() > 1.1:
            print(f"[WARN] ⚠ Actions outside expected range [-1, 1]")
        
        print(f"\n[SUCCESS] ✓ Policy produces variable actions")
        return True
    
    def _verify_observation_format(self, obs: dict):
        """Verify observation format matches what policy expects."""
        
        policy_obs = obs["policy"]
        
        print(f"\n{'='*80}")
        print(f"[OBS VERIFY] Checking observation format")
        print(f"{'='*80}")
        print(f"[OBS VERIFY] Observation shape: {policy_obs.shape}")
        print(f"[OBS VERIFY] Expected shape: ({self.num_envs}, {self.cfg.observation_space})")
        
        if policy_obs.shape[1] != self.cfg.observation_space:
            print(f"[ERROR] ❌ Observation size mismatch!")
            print(f"[ERROR] Expected: {self.cfg.observation_space}")
            print(f"[ERROR] Got: {policy_obs.shape[1]}")
            return False
        
        # Check for NaN/Inf
        if torch.isnan(policy_obs).any():
            print(f"[ERROR] ❌ Observation contains NaN!")
            nan_count = torch.isnan(policy_obs).sum().item()
            print(f"[ERROR] NaN count: {nan_count}")
            return False
        
        if torch.isinf(policy_obs).any():
            print(f"[ERROR] ❌ Observation contains Inf!")
            inf_count = torch.isinf(policy_obs).sum().item()
            print(f"[ERROR] Inf count: {inf_count}")
            return False
        
        # Check observation ranges
        obs_min = policy_obs[0].min().item()
        obs_max = policy_obs[0].max().item()
        obs_mean = policy_obs[0].mean().item()
        obs_std = policy_obs[0].std().item()
        
        print(f"\n[OBS VERIFY] Observation statistics (env 0):")
        print(f"  Min: {obs_min:.4f}")
        print(f"  Max: {obs_max:.4f}")
        print(f"  Mean: {obs_mean:.4f}")
        print(f"  Std: {obs_std:.4f}")
        
        # Check if point cloud is in observation
        pc_size = self.cfg.num_object_pc_points * 3
        pc_start_idx = 3 + 4 + 3 + 3 + 6 + 6 + 30 + 21  # Calculate start of PC in obs
        pc_obs = policy_obs[0, pc_start_idx:pc_start_idx+pc_size]
        
        pc_mean = pc_obs.mean().item()
        pc_magnitude = torch.norm(pc_obs.reshape(-1, 3), dim=1).mean().item()
        
        print(f"\n[OBS VERIFY] Point cloud in observation:")
        print(f"  Start index: {pc_start_idx}")
        print(f"  Size: {pc_size}")
        print(f"  Mean value: {pc_mean:.4f}")
        print(f"  Average magnitude: {pc_magnitude:.4f}")
        
        if pc_magnitude < 0.01:
            print(f"[ERROR] ❌ Point cloud in observation is too small!")
            print(f"[ERROR] This suggests point cloud is not properly transformed")
            return False
        
        if pc_magnitude > 10.0:
            print(f"[ERROR] ❌ Point cloud in observation is too large!")
            print(f"[ERROR] This suggests wrong scale or coordinate system")
            return False
        
        print(f"\n[SUCCESS] ✓ Observation format appears correct")
        return True



    def run_full_diagnostics(self):
        """Run all diagnostic checks when activating a new object."""
        
        if self._current_object_idx >= len(self._picking_order):
            return
        
        obj_idx = self._picking_order[self._current_object_idx]
        obj_info = self._object_infos[obj_idx]
        
        print(f"\n{'#'*80}")
        print(f"# RUNNING FULL DIAGNOSTICS FOR: {obj_info.object_id}")
        print(f"{'#'*80}\n")
        
        # Run all checks
        results = {
            'mesh': self._verify_mesh_extraction(obj_idx),
            'policy': self._verify_policy_object_match(obj_info),
            'observation': self._verify_observation_format(self._get_observations()),
            'policy_output': self._verify_policy_output()
        }
        
        print(f"\n{'#'*80}")
        print(f"# DIAGNOSTIC SUMMARY")
        print(f"{'#'*80}")
        
        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{check.upper():20s}: {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print(f"\n[SUCCESS] ✓✓✓ All diagnostics passed!")
        else:
            print(f"\n[ERROR] ✗✗✗ Some diagnostics failed!")
            print(f"[ERROR] Review the output above to identify issues")
        
        print(f"{'#'*80}\n")
        
        return all_passed


    def _compute_chamfer_distances(self, site_positions: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        """Compute minimum distance from each site to nearest point cloud point.
        Both inputs should be in WORLD coordinates."""
        site_positions = check_tensor_validity(site_positions, "site_positions", replace_invalid=True)
        point_cloud = check_tensor_validity(point_cloud, "point_cloud", replace_invalid=True)
        
        try:
            dists = torch.cdist(site_positions, point_cloud)
            dists = check_tensor_validity(dists, "cdist_result", replace_invalid=True)
        except RuntimeError as e:
            print(f"[ERROR] cdist failed: {e}")
            site_expanded = site_positions.unsqueeze(2)
            pc_expanded = point_cloud.unsqueeze(1)
            dists = torch.norm(site_expanded - pc_expanded, dim=-1)
            dists = check_tensor_validity(dists, "manual_dist_result", replace_invalid=True)
        
        chamfer_dists, _ = torch.min(dists, dim=2)
        chamfer_dists = check_tensor_validity(chamfer_dists, "chamfer_dists", replace_invalid=True)
        chamfer_dists = torch.clamp(chamfer_dists, 0.0, 10.0)
        
        return chamfer_dists

    def _get_contact_forces(self) -> torch.Tensor:
        """Extract contact forces using Isaac Sim's contact report API.
        
        Returns:
            torch.Tensor: (num_envs, 17) - Individual sphere contact forces
        """
        if not self._contact_sensors_enabled:
            return torch.zeros((self.num_envs, 17), device=self.device, dtype=torch.float32)
        
        try:
            
            # Get contact report interface
            contact_report_api = omni.physx.get_physx_simulation_interface()
            stage = omni.usd.get_context().get_stage()
            
            # Initialize force arrays
            finger_forces = torch.zeros((self.num_envs, 10), device=self.device, dtype=torch.float32)
            palm_forces = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
            
            # Query contact forces for each environment
            for env_idx in range(self.num_envs):
                robot_root = f"{self.scene.env_prim_paths[env_idx]}/Robot/root"
                
                # Query finger sphere forces
                for sphere_idx, sphere_rel_path in enumerate(self._finger_sphere_paths):
                    sphere_path = f"{robot_root}/{sphere_rel_path}"
                    prim = stage.GetPrimAtPath(sphere_path)
                    
                    if prim.IsValid():
                        # Get rigid body handle (parent link's rigid body)
                        # The sphere inherits forces from its parent
                        parent_path = "/".join(sphere_path.split("/")[:-2])  # Get parent link path
                        
                        # Query contact forces from PhysX
                        # Note: This reads from the simulation's contact buffer
                        contacts = contact_report_api.get_contact_report()
                        
                        total_force = 0.0
                        if contacts:
                            for contact in contacts:
                                # Check if this contact involves our sphere
                                if sphere_path in str(contact):
                                    # Extract force magnitude from impulse
                                    impulse = contact.get('impulse', (0, 0, 0))
                                    force_mag = (impulse[0]**2 + impulse[1]**2 + impulse[2]**2)**0.5
                                    total_force += force_mag
                        
                        finger_forces[env_idx, sphere_idx] = total_force
                
                # Query palm sphere forces (same process)
                for sphere_idx, sphere_rel_path in enumerate(self._palm_sphere_paths):
                    sphere_path = f"{robot_root}/{sphere_rel_path}"
                    prim = stage.GetPrimAtPath(sphere_path)
                    
                    if prim.IsValid():
                        contacts = contact_report_api.get_contact_report()
                        
                        total_force = 0.0
                        if contacts:
                            for contact in contacts:
                                if sphere_path in str(contact):
                                    impulse = contact.get('impulse', (0, 0, 0))
                                    force_mag = (impulse[0]**2 + impulse[1]**2 + impulse[2]**2)**0.5
                                    total_force += force_mag
                        
                        palm_forces[env_idx, sphere_idx] = total_force
            
            # Apply filtering
            finger_forces = torch.clamp(finger_forces, self.cfg.contact_force_range[0], self.cfg.contact_force_range[1])
            palm_forces = torch.clamp(palm_forces, self.cfg.contact_force_range[0], self.cfg.contact_force_range[1])
            
            # Store for debugging
            self._finger_contact_forces = finger_forces
            self._palm_contact_forces = palm_forces
            
            # Combine: (num_envs, 17)
            all_forces = torch.cat([finger_forces, palm_forces], dim=1)
            all_forces = torch.nan_to_num(all_forces, nan=0.0, posinf=self.cfg.max_contact_force, neginf=0.0)
            
            return all_forces
            
        except Exception as e:
            print(f"[WARN] PhysX contact query failed, using chamfer-based approximation: {e}")
            # Fall back to chamfer-based contact approximation
            return self._get_contact_forces_from_chamfer()

    def _debug_print_contact_forces(self):
        """Debug utility to print contact force statistics."""
        if not self._contact_sensors_enabled or self.num_envs == 0:
            return
        
        env_id = 0
        
        finger_forces = self._finger_contact_forces[env_id]
        palm_forces = self._palm_contact_forces[env_id]
        
        # Count active contacts
        active_fingers = (finger_forces > self.cfg.contact_force_threshold).sum().item()
        active_palm = (palm_forces > self.cfg.contact_force_threshold).sum().item()
        
        print("\n" + "="*60)
        print(f"[CONTACT DEBUG] Environment {env_id}")
        print("="*60)
        print(f"Active finger contacts: {active_fingers}/10")
        print(f"Active palm contacts: {active_palm}/7")
        
        # Print individual sensor readings
        print("\n[Finger Forces (N)]")
        finger_names = ["Thumb1", "Thumb2", "Index1", "Index2", "Middle1", 
                       "Middle2", "Ring1", "Ring2", "Pinky1", "Pinky2"]
        for i, name in enumerate(finger_names):
            force = finger_forces[i].item()
            active = "â" if force > self.cfg.contact_force_threshold else "â"
            print(f"  {name:8s}: {force:6.2f} N {active}")
        
        print("\n[Palm Forces (N)]")
        palm_names = ["A1", "A2", "A3", "B2", "B3", "C2", "C3"]
        for i, name in enumerate(palm_names):
            force = palm_forces[i].item()
            active = "â" if force > self.cfg.contact_force_threshold else "â"
            print(f"  {name:4s}: {force:6.2f} N {active}")
        
        print(f"\nTotal force: {(finger_forces.sum() + palm_forces.sum()).item():.2f} N")
        print("="*60 + "\n")

    def _get_site_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WORLD positions of finger and palm contact sensor sites."""
        
        if not self._sites_initialized:
            if not hasattr(self, '_site_warning_printed'):
                print("[WARN] Sites not initialized! Returning zeros.")
                self._site_warning_printed = True
            finger_site_pos = torch.zeros((self.num_envs, 10, 3), device=self.device)
            palm_site_pos = torch.zeros((self.num_envs, 7, 3), device=self.device)
            return finger_site_pos, palm_site_pos
        
        # Get parent body states (WORLD coordinates)
        all_body_pos = self.robot.data.body_pos_w
        all_body_quat = self.robot.data.body_quat_w
        
        # Validate body data
        if torch.isnan(all_body_pos).any() or torch.isnan(all_body_quat).any():
            print("[ERROR] NaN detected in body positions/orientations!")
            return torch.zeros((self.num_envs, 10, 3), device=self.device), \
                   torch.zeros((self.num_envs, 7, 3), device=self.device)
        
        # Get finger parent positions and orientations
        finger_parent_pos = all_body_pos[:, self._finger_parent_idx, :]
        finger_parent_quat = all_body_quat[:, self._finger_parent_idx, :]
        
        # Transform local site positions to world coordinates
        finger_site_pos = finger_parent_pos + quat_apply_vec(
            finger_parent_quat, 
            self._finger_site_local_pos.unsqueeze(0).expand(self.num_envs, -1, -1)
        )
        
        # Get palm parent positions and orientations  
        palm_parent_pos = all_body_pos[:, self._palm_parent_idx, :]
        palm_parent_quat = all_body_quat[:, self._palm_parent_idx, :]
        
        palm_site_pos = palm_parent_pos + quat_apply_vec(
            palm_parent_quat,
            self._palm_site_local_pos.unsqueeze(0).expand(self.num_envs, -1, -1)
        )
        
        return finger_site_pos, palm_site_pos

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions.clone()

    def _reset_hand_above_target_object(self, obj_idx: int):
        """Reset hand to starting position above target object.
        
        Used after drops to retry picking.
        
        Args:
            obj_idx: Always 0 (target object)
        """
        target_object = self.objects[obj_idx]
        target_object.update(dt=self.cfg.sim.dt)
        
        target_pos = target_object.data.root_pos_w
        
        # ── Sync object_init_pos to the current settled position ─────────────
        # In isolated mode the object can spawn at a random XY; if object_init_pos
        # was captured before physics settled it won't match the actual resting
        # position, which corrupts the policy's approach direction signal.
        # We always refresh it here so both the first reset and every inter-trial
        # respawn record the true post-settle position.
        self.object_init_pos = target_pos.clone()
        # ─────────────────────────────────────────────────────────────────────
        
        # Position hand above target
        hand_pos = target_pos.clone()
        hand_pos[:, 2] += self.cfg.hand_init_height_above_object
        
        # Reset hand state
        hand_state = self.robot.data.default_root_state.clone()
        hand_state[:, 0:3] = hand_pos
        
        # Reset orientation
        base_roll = -np.pi
        euler = torch.zeros((self.num_envs, 3), device=self.device)
        euler[:, 0] = base_roll
        
        quat = euler_to_quaternion(euler)
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        hand_state[:, 3:7] = quat
        hand_state[:, 7:] = 0.0
        
        self.robot.write_root_state_to_sim(hand_state)
        
        # Reset fingers
        all_joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(all_joint_pos)

        self.robot.write_joint_state_to_sim(all_joint_pos, joint_vel, None)
        
        # Reset velocities
        self.pos_velocity[:] = 0.0
        self.rot_velocity[:] = 0.0
        
        # Let physics settle
        for _ in range(20):
            self.sim.step(render=False)
        
        self.robot.update(dt=self.cfg.sim.dt)
        
        print(f"[RESET] Hand repositioned above target after drop")
        
    def _apply_action(self) -> None:
        """Apply actions using velocity control that respects physics."""
        
        # Decode actions with clipping
        pos_actions = torch.clamp(self.actions[:, 0:3], -1.0, 1.0)
        rot_actions = torch.clamp(self.actions[:, 3:6], -1.0, 1.0)
        finger_actions = torch.clamp(self.actions[:, 6:12], -1.0, 1.0)
        
        # Convert to TARGET VELOCITIES (not position deltas!)
        # Scale to reasonable velocities: 0.2 m/s linear, 0.5 rad/s angular
        target_lin_vel = pos_actions * self.cfg.max_pos_delta
        target_ang_vel = rot_actions * self.cfg.max_rot_delta
        
        # Apply smoothing to velocities
        self.pos_velocity = (self.cfg.hand_velocity_smoothing * self.pos_velocity + 
                        (1 - self.cfg.hand_velocity_smoothing) * target_lin_vel)
        self.rot_velocity = (self.cfg.hand_velocity_smoothing * self.rot_velocity + 
                        (1 - self.cfg.hand_velocity_smoothing) * target_ang_vel)
        
        # Get current state (will be updated in next _get_observations call)
        current_pos = self.robot.data.root_pos_w
        current_quat = self.robot.data.root_quat_w
        current_euler = quaternion_to_euler(current_quat)
        
        # ============ FIXED: Clamp rotation to ±90° from REFERENCE ============
        # Use reference orientation from config (already initialized in __init__)
        # self.ref_palm_euler = (π, 0, 0) from cfg.ref_palm_euler
        
        # Compute angular deviation from reference (signed difference in [-π, π])
        euler_diff = wrap_angle_diff_for_limits(current_euler, self.ref_palm_euler)
        
        rotation_limit = self.cfg.rot_limit_rad  # π/2 = 90 degrees
        
        # Zero out angular velocity components that would exceed limits
        # Roll (x-axis) - deviation from reference roll
        roll_at_pos_limit = (euler_diff[:, 0] >= rotation_limit) & (self.rot_velocity[:, 0] > 0)
        roll_at_neg_limit = (euler_diff[:, 0] <= -rotation_limit) & (self.rot_velocity[:, 0] < 0)
        self.rot_velocity[:, 0] = torch.where(
            roll_at_pos_limit | roll_at_neg_limit, 
            torch.zeros_like(self.rot_velocity[:, 0]), 
            self.rot_velocity[:, 0]
        )
        
        # Pitch (y-axis) - deviation from reference pitch
        pitch_at_pos_limit = (euler_diff[:, 1] >= rotation_limit) & (self.rot_velocity[:, 1] > 0)
        pitch_at_neg_limit = (euler_diff[:, 1] <= -rotation_limit) & (self.rot_velocity[:, 1] < 0)
        self.rot_velocity[:, 1] = torch.where(
            pitch_at_pos_limit | pitch_at_neg_limit,
            torch.zeros_like(self.rot_velocity[:, 1]),
            self.rot_velocity[:, 1]
        )
        
        # Yaw (z-axis) - deviation from reference yaw
        yaw_at_pos_limit = (euler_diff[:, 2] >= rotation_limit) & (self.rot_velocity[:, 2] > 0)
        yaw_at_neg_limit = (euler_diff[:, 2] <= -rotation_limit) & (self.rot_velocity[:, 2] < 0)
        self.rot_velocity[:, 2] = torch.where(
            yaw_at_pos_limit | yaw_at_neg_limit,
            torch.zeros_like(self.rot_velocity[:, 2]),
            self.rot_velocity[:, 2]
        )
        # =====================================================================
        
        # Apply position limits by zeroing velocity when at boundaries
        env_origins = self.scene.env_origins
        pos_rel = current_pos - env_origins
        
        # Soft boundary damping (gradual slowdown near limits)
        margin = 0.05

        # X damping
        x_factor = torch.ones(self.num_envs, device=self.device)
        x_over = torch.abs(pos_rel[:, 0]) - (self.cfg.hand_x_limit - margin)
        x_factor = torch.where(x_over > 0, torch.clamp(1.0 - x_over/margin, 0.1, 1.0), x_factor)
        self.pos_velocity[:, 0] *= x_factor
        
        # Y damping
        y_factor = torch.ones(self.num_envs, device=self.device)
        y_over = torch.abs(pos_rel[:, 1]) - (self.cfg.hand_y_limit - margin)
        y_factor = torch.where(y_over > 0, torch.clamp(1.0 - y_over/margin, 0.1, 1.0), y_factor)
        self.pos_velocity[:, 1] *= y_factor
        
        # Z damping
        z_factor = torch.ones(self.num_envs, device=self.device)
        z_under = self.cfg.hand_z_min - pos_rel[:, 2]
        z_over = pos_rel[:, 2] - self.cfg.hand_z_max
        z_factor = torch.where(z_under > 0, torch.clamp(1.0 - z_under/margin, 0.1, 1.0), z_factor)
        z_factor = torch.where(z_over > 0, torch.clamp(1.0 - z_over/margin, 0.1, 1.0), z_factor)
        self.pos_velocity[:, 2] *= z_factor

        # CRITICAL: Clamp velocities to prevent instability
        self.pos_velocity = torch.clamp(self.pos_velocity, -1.0, 1.0)
        self.rot_velocity = torch.clamp(self.rot_velocity, -2.0, 2.0)
        
        # Write velocities to simulation
        root_velocities = self.robot.data.root_state_w.clone()
        root_velocities[:, 7:10] = self.pos_velocity
        root_velocities[:, 10:13] = self.rot_velocity
        
        self.robot.write_root_velocity_to_sim(root_velocities[:, 7:13])
        
        # ============ CRITICAL FIX: Don't read position here - will be updated in _get_observations ============
        # These will be updated at the start of next _get_observations() call after physics step
        # Removed these lines:
        # self.current_pos = self.robot.data.root_pos_w.clone()
        # self.current_euler = quaternion_to_euler(self.robot.data.root_quat_w)
        # ===================================================================================================
        
        # Finger control (position control is fine for joints)
        finger_delta = finger_actions * self.cfg.max_finger_delta
        new_joint_pos = self.current_joint_pos + finger_delta
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self._actuated_joint_idx]
        lower_limits = joint_limits[:, :, 0]
        upper_limits = joint_limits[:, :, 1]
        new_joint_pos = torch.clamp(new_joint_pos, lower_limits, upper_limits)
        
        new_joint_pos = check_tensor_validity(new_joint_pos, "new_joint_pos", replace_invalid=True)
        self.current_joint_pos = new_joint_pos

        all_joint_pos = self.robot.data.default_joint_pos.clone()
        all_joint_pos[:, self._actuated_joint_idx] = self.current_joint_pos

        self.robot.set_joint_position_target(all_joint_pos)
        self.robot.write_data_to_sim()
        
        # Update visualizations
        if self.cfg.visualize_point_cloud:
            self._update_point_cloud_visualization()
            self._update_site_visualization()
            self._update_object_root_visualization()

    def _get_observations(self) -> dict:
        """Get observations for the policy with full scene synchronization.
        
        Ensures ALL objects in cluttered scene are updated before reading observations.
        
        NOTE: For cluttered scene, we only observe the TARGET object's point cloud,
        not the other objects on the table. The policy must learn to avoid them
        through collision/contact feedback.
        """

        # Safety check: if all objects done, return zero observations
        if self._current_object_idx >= len(self._picking_order):
            obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
            return {"policy": obs}
        
        # ============ CRITICAL FIX: Update ALL objects in cluttered scene ============
        # This ensures collision detection and physics state are current
        for obj in self.objects:
            obj.update(dt=self.cfg.sim.dt)
        
        # Update robot state
        self.robot.update(dt=self.cfg.sim.dt)
        
        # NOW update tracking variables with fresh data
        self.current_pos = self.robot.data.root_pos_w.clone()
        self.current_euler = quaternion_to_euler(self.robot.data.root_quat_w)
        # ============================================================================
        
        # Get current target object (now guaranteed to be synced)
        obj_idx = self._picking_order[self._current_object_idx]
        current_object = self.objects[obj_idx]
        
        # Get hand state (already updated above)
        hand_pos_world = self.robot.data.root_pos_w.clone()
        hand_quat = self.robot.data.root_quat_w.clone()
        hand_lin_vel = self.robot.data.root_lin_vel_w.clone()
        hand_ang_vel = self.robot.data.root_ang_vel_w.clone()
        
        # Convert to RELATIVE
        hand_pos = hand_pos_world - self.scene.env_origins
        
        # Joint states
        joint_pos = self.robot.data.joint_pos[:, self._actuated_joint_idx].clone()
        joint_vel = self.robot.data.joint_vel[:, self._actuated_joint_idx].clone()
        
        # Site positions (these depend on robot state which we just updated)
        finger_site_pos_world, palm_site_pos_world = self._get_site_positions()
        finger_site_pos = finger_site_pos_world - self.scene.env_origins.unsqueeze(1)
        palm_site_pos = palm_site_pos_world - self.scene.env_origins.unsqueeze(1)
        finger_site_pos_flat = finger_site_pos.reshape(self.num_envs, -1)
        palm_site_pos_flat = palm_site_pos.reshape(self.num_envs, -1)
        
        # TARGET object state (uses freshly updated state)
        object_pos_world = current_object.data.root_pos_w.clone()
        object_quat = current_object.data.root_quat_w.clone()
        object_lin_vel = current_object.data.root_lin_vel_w.clone()
        object_ang_vel = current_object.data.root_ang_vel_w.clone()
        
        object_pos = object_pos_world - self.scene.env_origins
        
        # Contact forces (empty if disabled)
        contact_forces = torch.zeros((self.num_envs, 17), device=self.device)
        
        # Sample TARGET object point cloud (this calls _sample_object_point_cloud which does its own update)
        object_pc_world = self._sample_object_point_cloud()
        object_pc_rel = object_pc_world - self.scene.env_origins.unsqueeze(1)
        object_pc_flat = object_pc_rel.reshape(self.num_envs, -1)

        # Debug verification
        if self._debug_step_counter % 100 == 0 and self.cfg.enable_debug_obs:
            obj_id = self._object_infos[obj_idx].object_id
            
            print(f"\n[OBS DEBUG] Step {self._debug_step_counter}")
            print(f"[OBS DEBUG] Target object: {obj_id} (index {obj_idx})")
            print(f"[OBS DEBUG] Object position (world): [{object_pos_world[0, 0]:.3f}, {object_pos_world[0, 1]:.3f}, {object_pos_world[0, 2]:.3f}]")
            print(f"[OBS DEBUG] Object position (relative): [{object_pos[0, 0]:.3f}, {object_pos[0, 1]:.3f}, {object_pos[0, 2]:.3f}]")
            print(f"[OBS DEBUG] Hand position (relative): [{hand_pos[0, 0]:.3f}, {hand_pos[0, 1]:.3f}, {hand_pos[0, 2]:.3f}]")
            print(f"[OBS DEBUG] Hand-to-object distance: {torch.norm(hand_pos_world[0] - object_pos_world[0]).item():.3f}m")
            pc_center_world = object_pc_world[0].mean(dim=0)
            print(f"[OBS DEBUG] Point cloud center (world): [{pc_center_world[0]:.3f}, {pc_center_world[1]:.3f}, {pc_center_world[2]:.3f}]")
            pc_to_obj = torch.norm(pc_center_world - object_pos_world[0]).item()
            print(f"[OBS DEBUG] Point cloud to object distance: {pc_to_obj:.3f}m")
            if pc_to_obj > 0.05:
                print(f"[ERROR] ⚠️⚠️⚠️ POINT CLOUD MISALIGNED WITH OBJECT! ⚠️⚠️⚠️")
        
        # Chamfer distances to TARGET object
        all_sites_world = torch.cat([finger_site_pos_world, palm_site_pos_world], dim=1)
        chamfer_distances = self._compute_chamfer_distances(all_sites_world, object_pc_world)
        
        # Object initial position (stored at reset)
        object_init_pos_rel = self.object_init_pos - self.scene.env_origins
        
        # ── Base obs (234 dims, identical to training base env) ───────────────
        obs = torch.cat([
            hand_pos,               # 3
            hand_quat,              # 4
            hand_lin_vel,           # 3
            hand_ang_vel,           # 3
            joint_pos,              # 6
            joint_vel,              # 6
            finger_site_pos_flat,   # 30
            palm_site_pos_flat,     # 21
            object_pc_flat,         # num_object_pc_points * 3
            chamfer_distances,      # 17
            contact_forces,         # 17
            object_pos,             # 3
            object_quat,            # 4
            object_lin_vel,         # 3
            object_ang_vel,         # 3
            object_init_pos_rel,    # 3
            self.actions            # 12
        ], dim=-1)
        # Total base: 234

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on task performance.
        Uses WORLD coordinates for all computations."""

        # Safety check: if all objects done, return zero reward
        if self._current_object_idx >= len(self._picking_order):
            return torch.zeros(self.num_envs, device=self.device)
        
        hand_pos_world = self.robot.data.root_pos_w

        # TARGET object
        obj_idx = self._picking_order[self._current_object_idx]
        current_object = self.objects[obj_idx]
        object_pos_world = current_object.data.root_pos_w
        object_init_pos_world = self.object_init_pos
        
        hand_quat = self.robot.data.root_quat_w
        palm_euler = quaternion_to_euler(hand_quat)
        
        joint_pos = self.robot.data.joint_pos[:, self._actuated_joint_idx]
        
        # Get chamfer distances (using WORLD coordinates)
        finger_site_pos_world, palm_site_pos_world = self._get_site_positions()
        all_sites_world = torch.cat([finger_site_pos_world, palm_site_pos_world], dim=1)
        
        object_pc_world = self._sample_object_point_cloud()
        chamfer_distances = self._compute_chamfer_distances(all_sites_world, object_pc_world)
        
        contact_forces = self._get_contact_forces()

        return compute_rewards_shaped(
            hand_pos=hand_pos_world,
            object_pos=object_pos_world,
            object_init_pos=object_init_pos_world,
            palm_euler=palm_euler,
            joint_pos=joint_pos,
            chamfer_distances=chamfer_distances,
            contact_forces=contact_forces,
            actions=self.actions,
            reset_terminated=self.reset_terminated,
            object_lifted=self.object_lifted,
            cfg=self.cfg,
            env_origins=self.scene.env_origins,
        )
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination with proper chaos tracking for ALL trials including last.
        
        MODIFIED: Hand does NOT reset as long as object is lifted above min_lift_height.
        FIXED: Ensures positions are captured for ALL trials, including trial 10.
        ADDED: Hard trial timeout — if _trial_step_counter >= max_trial_timesteps,
            the trial is marked as failed (reason='timeout') and skipped entirely.
            This is SEPARATE from max_steps_per_trial (which only respawns the hand).
        """

        # Increment trial step counter
        self._trial_step_counter += 1

        # Initialize trial start time
        if self._trial_start_time is None:
            self._trial_start_time = time.time()
            self._picking_start_step = self.common_step_counter

        # ── Pre-fetch target object (needed by both timeout and main logic) ────────
        target_object = self.objects[0]
        target_object.update(dt=self.cfg.sim.dt)
        target_pos_world = target_object.data.root_pos_w[0].cpu()

        # ══════════════════════════════════════════════════════════════════════════
        # HARD TRIAL TIMEOUT
        # Completely separate from max_steps_per_trial (hand respawn).
        # max_steps_per_trial  → only respawns the hand, trial continues
        # max_trial_timesteps  → fails the entire trial, moves to next one
        # ══════════════════════════════════════════════════════════════════════════
        max_trial_timesteps = getattr(self.cfg, 'max_trial_timesteps', 100_000)

        if self._trial_step_counter >= max_trial_timesteps:
            print(
                f"\n[TIMEOUT] Trial {self._current_trial + 1}/{self.cfg.num_trials} "
                f"reached hard limit of {max_trial_timesteps} steps — marking as FAILED"
            )

            # Capture final position using last known on-table position
            if 0 not in self.target_last_table_pos:
                self.target_last_table_pos[0] = (
                    float(target_pos_world[0].item()),
                    float(target_pos_world[1].item()),
                )
            if self.target_final_pos is None:
                self.target_final_pos = self.target_last_table_pos.copy()

            # Safety fallbacks for chaos calculation
            if self.target_initial_pos is None or 0 not in self.target_initial_pos:
                print(f"[TIMEOUT] No initial position recorded — using (0, 0) fallback")
                self.target_initial_pos = {0: (0.0, 0.0)}
            if self.target_final_pos is None or 0 not in self.target_final_pos:
                self.target_final_pos = {0: (0.0, 0.0)}

            picking_time = time.time() - self._trial_start_time

            chaos_metrics = calculate_scene_chaos(
                initial_positions=self.target_initial_pos.copy(),
                final_positions=self.target_final_pos.copy(),
                object_status={0: 'timeout'},
            )

            trial_result = {
                'trial':         self._current_trial + 1,
                'success':       False,
                'drops':         self._current_object_drops,
                'hand_respawns': self._hand_respawn_count,
                'steps':         self._trial_step_counter,
                'picking_time':  picking_time,
                'reason':        'timeout',
                'chaos_metrics': chaos_metrics,
            }
            self._trial_results.append(trial_result)
            self._trial_final_positions.append(self.target_final_pos.copy())

            print(f"[TIMEOUT] Drops: {self._current_object_drops}  "
                f"Hand respawns: {self._hand_respawn_count}  "
                f"Time: {picking_time:.1f}s  "
                f"Chaos: {chaos_metrics['target_distance']:.4f}m")

            self._current_trial += 1

            if self._current_trial >= self.cfg.num_trials:
                print(f"[COMPLETE] All {self.cfg.num_trials} trials finished "
                    f"(last trial ended via timeout).")
                self._state = "COMPLETE"
                terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                return terminated, time_out

            # Start next trial
            print(f"\n[TRIAL {self._current_trial + 1}] Starting after timeout...")
            self._respawn_object_for_new_trial()
            self._reset_hand_above_target_object(0)

            # Reset ALL per-trial counters
            self._current_object_drops  = 0
            self._hand_respawn_count    = 0
            self._trial_step_counter    = 0          # ← also resets hand-respawn counter
            self.target_final_pos       = None
            self._trial_start_time      = time.time()
            self._picking_start_step    = self.common_step_counter
            self._state                 = "PICKING"

            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return terminated, time_out
        # ══════════════════════════════════════════════════════════════════════════
        # END HARD TRIAL TIMEOUT
        # Everything below this line is the original per-step logic unchanged.
        # ══════════════════════════════════════════════════════════════════════════

        # ── Current height / XY tracking (uses already-updated target_object) ─────
        table_height    = self.scene.env_origins[0, 2] + self.cfg.table_height
        current_height  = float(target_pos_world[2].item())
        height_above_table = current_height - table_height

        object_is_lifted = height_above_table >= self.cfg.min_lift_height

        current_xy = (
            float(target_pos_world[0].item()),
            float(target_pos_world[1].item()),
        )

        if current_height < (table_height + 0.001):
            self.target_last_table_pos[0] = current_xy
        elif 0 not in self.target_last_table_pos:
            self.target_last_table_pos[0] = current_xy

        # ── Hand-respawn checks (ONLY when object is NOT lifted) ──────────────────
        # These use max_steps_per_trial, which is SEPARATE from max_trial_timesteps.
        # Hitting this limit respawns the hand but the trial carries on.
        if object_is_lifted:
            if self._trial_step_counter % 50 == 0:
                print(f"[LIFTED] Object at {height_above_table:.3f}m — hand reset disabled")
        else:
            # Check hand flip
            if self.cfg.respawn_hand_on_flip and self._check_hand_flip():
                if not self._check_pick_success():
                    print(f"[HAND FLIP] Respawning hand (flip #{self._hand_respawn_count + 1})")
                    self._hand_respawn_count += 1
                    self._reset_hand_above_target_object(0)

                    terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    return terminated, time_out

            # Check per-trial hand-respawn step limit (NOT the hard timeout)
            if self._trial_step_counter % self.cfg.max_steps_per_trial == 0 \
                    and self._trial_step_counter > 0:
                if not self._check_pick_success():
                    print(
                        f"[HAND RESPAWN] {self._trial_step_counter} steps reached — "
                        f"respawning hand (respawn #{self._hand_respawn_count + 1}). "
                        f"Trial continues (hard limit: {max_trial_timesteps} steps)."
                    )
                    self._hand_respawn_count += 1
                    self._reset_hand_above_target_object(0)
                    # NOTE: _trial_step_counter is NOT reset here so the hard
                    #       timeout above keeps counting toward max_trial_timesteps.

                    terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    return terminated, time_out

        # ── State machine ─────────────────────────────────────────────────────────

        if self._state == "PICKING":
            target_obj_info = self._object_infos[0]

            if self._check_pick_success():
                print(f"\n[SUCCESS] Trial {self._current_trial + 1}/{self.cfg.num_trials} "
                    f"— {target_obj_info.object_id} lifted!")

                if self.target_final_pos is None:
                    if 0 not in self.target_last_table_pos:
                        self.target_last_table_pos[0] = current_xy
                    self.target_final_pos = self.target_last_table_pos.copy()

                    init_x, init_y   = self.target_initial_pos[0]
                    final_x, final_y = self.target_final_pos[0]
                    movement = np.sqrt((final_x - init_x)**2 + (final_y - init_y)**2)
                    print(f"[CHAOS] Initial: ({init_x:.4f}, {init_y:.4f})")
                    print(f"[CHAOS] Final:   ({final_x:.4f}, {final_y:.4f})")
                    print(f"[CHAOS] Movement: {movement:.4f}m")

                self._state = "REMOVING"
                self._removal_timer = 0

        elif self._state == "REMOVING":
            if self._check_pick_success():
                self._removal_timer += 1

                if self._removal_timer >= self.cfg.removal_delay_steps:
                    # ── TRIAL COMPLETE — SUCCESS ───────────────────────────────

                    if self.target_final_pos is None:
                        if 0 not in self.target_last_table_pos:
                            self.target_last_table_pos[0] = current_xy
                        self.target_final_pos = self.target_last_table_pos.copy()

                    if self.target_initial_pos is None or 0 not in self.target_initial_pos:
                        print(f"[ERROR] Trial {self._current_trial + 1}: No initial position! Using fallback.")
                        self.target_initial_pos = {0: (0.0, 0.0)}
                    if self.target_final_pos is None or 0 not in self.target_final_pos:
                        print(f"[ERROR] Trial {self._current_trial + 1}: No final position! Using fallback.")
                        self.target_final_pos = {0: (0.0, 0.0)}

                    picking_time  = time.time() - self._trial_start_time
                    chaos_metrics = calculate_scene_chaos(
                        initial_positions=self.target_initial_pos.copy(),
                        final_positions=self.target_final_pos.copy(),
                        object_status={0: 'success'},
                    )

                    trial_result = {
                        'trial':         self._current_trial + 1,
                        'success':       True,
                        'drops':         self._current_object_drops,
                        'hand_respawns': self._hand_respawn_count,
                        'steps':         self._trial_step_counter,
                        'picking_time':  picking_time,
                        'reason':        'success',
                        'chaos_metrics': chaos_metrics,
                    }
                    self._trial_results.append(trial_result)
                    self._trial_final_positions.append(self.target_final_pos.copy())

                    print(f"\n{'='*80}")
                    print(f"[TRIAL {self._current_trial + 1}] ✓✓✓ SUCCESS")
                    print(f"{'='*80}")
                    print(f"  Drops:         {self._current_object_drops}")
                    print(f"  Hand Respawns: {self._hand_respawn_count}")
                    print(f"  Steps:         {self._trial_step_counter}")
                    print(f"  Time:          {picking_time:.2f}s")
                    if chaos_metrics:
                        print(f"  Chaos: {chaos_metrics['target_distance']:.4f}m "
                            f"({chaos_metrics['target_normalized_distance']:.4f} normalized)")
                    print(f"{'='*80}\n")

                    self._current_trial += 1

                    if self._current_trial >= self.cfg.num_trials:
                        print(f"[COMPLETE] All {self.cfg.num_trials} trials finished!")
                        self._state = "COMPLETE"
                        terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                        time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                        return terminated, time_out

                    print(f"\n[TRIAL {self._current_trial + 1}] Starting new trial...")
                    self._respawn_object_for_new_trial()
                    self._reset_hand_above_target_object(0)

                    self._current_object_drops  = 0
                    self._hand_respawn_count    = 0
                    self._trial_step_counter    = 0
                    self.target_final_pos       = None
                    self._trial_start_time      = time.time()
                    self._picking_start_step    = self.common_step_counter
                    self._state                 = "PICKING"

            else:
                # Object fell back down after being lifted
                self._current_object_drops += 1
                print(f"\n[DROP] Trial {self._current_trial + 1} — fell back down "
                    f"(drop #{self._current_object_drops})")

                if self._current_object_drops >= self.cfg.max_drop_attempts:
                    # ── TRIAL FAILED — MAX DROPS ───────────────────────────────

                    if self.target_final_pos is None:
                        if 0 not in self.target_last_table_pos:
                            self.target_last_table_pos[0] = current_xy
                        self.target_final_pos = self.target_last_table_pos.copy()

                    if self.target_initial_pos is None or 0 not in self.target_initial_pos:
                        print(f"[ERROR] Trial {self._current_trial + 1}: No initial position! Using fallback.")
                        self.target_initial_pos = {0: (0.0, 0.0)}
                    if self.target_final_pos is None or 0 not in self.target_final_pos:
                        print(f"[ERROR] Trial {self._current_trial + 1}: No final position! Using fallback.")
                        self.target_final_pos = {0: (0.0, 0.0)}

                    picking_time  = time.time() - self._trial_start_time
                    chaos_metrics = calculate_scene_chaos(
                        initial_positions=self.target_initial_pos.copy(),
                        final_positions=self.target_final_pos.copy(),
                        object_status={0: 'failed'},
                    )

                    trial_result = {
                        'trial':         self._current_trial + 1,
                        'success':       False,
                        'drops':         self._current_object_drops,
                        'hand_respawns': self._hand_respawn_count,
                        'steps':         self._trial_step_counter,
                        'picking_time':  picking_time,
                        'reason':        'max_drops',
                        'chaos_metrics': chaos_metrics,
                    }
                    self._trial_results.append(trial_result)
                    self._trial_final_positions.append(self.target_final_pos.copy())

                    print(f"\n{'='*80}")
                    print(f"[TRIAL {self._current_trial + 1}] ✗✗✗ FAILED (max drops)")
                    print(f"{'='*80}")
                    if chaos_metrics:
                        print(f"  Chaos: {chaos_metrics['target_distance']:.4f}m")
                    print(f"{'='*80}\n")

                    self._current_trial += 1

                    if self._current_trial >= self.cfg.num_trials:
                        print(f"[COMPLETE] All {self.cfg.num_trials} trials finished!")
                        self._state = "COMPLETE"
                        terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                        time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                        return terminated, time_out

                    print(f"\n[TRIAL {self._current_trial + 1}] Starting new trial...")
                    self._respawn_object_for_new_trial()
                    self._reset_hand_above_target_object(0)

                    self._current_object_drops  = 0
                    self._hand_respawn_count    = 0
                    self._trial_step_counter    = 0
                    self.target_final_pos       = None
                    self._trial_start_time      = time.time()
                    self._picking_start_step    = self.common_step_counter
                    self._state                 = "PICKING"

                else:
                    # Still have drop attempts left — keep trying
                    self._state         = "PICKING"
                    self._removal_timer = 0

        elif self._state == "COMPLETE":
            terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return terminated, time_out

        # ── Check if target fell off table (only while PICKING) ───────────────────
        if self._state == "PICKING":
            object_pos_world    = target_object.data.root_pos_w
            env_origins         = self.scene.env_origins
            table_surface_world = env_origins[:, 2] + self.cfg.table_height
            object_fell_off     = object_pos_world[:, 2] < (table_surface_world - self.cfg.object_fall_margin)

            if object_fell_off.any():
                if self.target_final_pos is None:
                    if 0 not in self.target_last_table_pos:
                        self.target_last_table_pos[0] = current_xy
                    self.target_final_pos = self.target_last_table_pos.copy()

                if self.target_initial_pos is None or 0 not in self.target_initial_pos:
                    print(f"[ERROR] Trial {self._current_trial + 1}: No initial position! Using fallback.")
                    self.target_initial_pos = {0: (0.0, 0.0)}
                if self.target_final_pos is None or 0 not in self.target_final_pos:
                    print(f"[ERROR] Trial {self._current_trial + 1}: No final position! Using fallback.")
                    self.target_final_pos = {0: (0.0, 0.0)}

                picking_time  = time.time() - self._trial_start_time
                chaos_metrics = calculate_scene_chaos(
                    initial_positions=self.target_initial_pos.copy(),
                    final_positions=self.target_final_pos.copy(),
                    object_status={0: 'fell_off'},
                )

                trial_result = {
                    'trial':         self._current_trial + 1,
                    'success':       False,
                    'drops':         self._current_object_drops,
                    'hand_respawns': self._hand_respawn_count,
                    'steps':         self._trial_step_counter,
                    'picking_time':  picking_time,
                    'reason':        'fell_off_table',
                    'chaos_metrics': chaos_metrics,
                }
                self._trial_results.append(trial_result)
                self._trial_final_positions.append(self.target_final_pos.copy())

                print(f"\n{'='*80}")
                print(f"[TRIAL {self._current_trial + 1}] ✗✗✗ FAILED (fell off table)")
                print(f"{'='*80}")
                if chaos_metrics:
                    print(f"  Chaos: {chaos_metrics['target_distance']:.4f}m")
                print(f"{'='*80}\n")

                self._current_trial += 1

                if self._current_trial >= self.cfg.num_trials:
                    print(f"[COMPLETE] All {self.cfg.num_trials} trials finished!")
                    self._state = "COMPLETE"
                    terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                    time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    return terminated, time_out

                print(f"\n[TRIAL {self._current_trial + 1}] Starting new trial...")
                self._respawn_object_for_new_trial()
                self._reset_hand_above_target_object(0)

                self._current_object_drops  = 0
                self._hand_respawn_count    = 0
                self._trial_step_counter    = 0
                self.target_final_pos       = None
                self._trial_start_time      = time.time()
                self._picking_start_step    = self.common_step_counter
                self._state                 = "PICKING"

        # Still running — no termination this step
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environment - position hand above TARGET object only.
        
        Target object is always at index 0.
        Clutter objects remain on table (not reset).
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = env_ids
        
        if len(env_ids_tensor) == 0:
            return
        
        self.total_episodes[env_ids_tensor] += 1 
        super()._reset_idx(env_ids_tensor)
        
        num_resets = len(env_ids_tensor)
        env_origins = self.scene.env_origins[env_ids_tensor]
        
        # Get TARGET object position (index 0)
        target_object = self.objects[0]
        target_object.update(dt=self.cfg.sim.dt)
        
        # ── In isolated mode the object was placed then physics-settled in
        # _respawn_object_for_new_trial / _setup_scene before _reset_idx is called.
        # super()._reset_idx() may drive the sim one extra step, so we re-read
        # the object position AFTER that call to get the true settled position.
        # This prevents the policy from receiving a stale object_init_pos that
        # doesn't match the actual post-settle location (most visible in isolated
        # mode where the object can spawn off-centre and drift slightly).
        target_obj_pos = target_object.data.root_pos_w[env_ids_tensor]
        
        # If the object is at its default (pre-settle) height, run a quick settle
        # to get the true resting position before locking in object_init_pos.
        # We only do this on the very first reset (common_step_counter == 0) or
        # when in isolated mode, since cluttered mode always spawns at centre.
        if self.cfg.use_isolated_mode:
            # Run a few extra physics steps to make sure object has settled
            for _ in range(10):
                self.sim.step(render=False)
            target_object.update(dt=self.cfg.sim.dt)
            target_obj_pos = target_object.data.root_pos_w[env_ids_tensor]
        
        # Store initial position for reward computation
        self.object_init_pos[env_ids_tensor] = target_obj_pos
        
        # Position hand above target
        hand_pos_world = target_obj_pos.clone()
        hand_pos_world[:, 2] += self.cfg.hand_init_height_above_object
        
        self.current_pos[env_ids_tensor] = hand_pos_world
        
        # Reset orientation
        base_roll = -np.pi
        self.current_euler[env_ids_tensor, 0] = base_roll
        self.current_euler[env_ids_tensor, 1] = 0.0
        self.current_euler[env_ids_tensor, 2] = 0.0
        
        quat = euler_to_quaternion(self.current_euler[env_ids_tensor])
        quat = quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-6)
        
        root_state = self.robot.data.default_root_state[env_ids_tensor].clone()
        root_state[:, 0:3] = hand_pos_world
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0
        
        # Reset fingers
        all_joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = torch.zeros_like(all_joint_pos)

        self.current_joint_pos[env_ids_tensor] = all_joint_pos[:, self._actuated_joint_idx]
        
        # Reset tracking
        self.pos_velocity[env_ids_tensor] = 0.0
        self.rot_velocity[env_ids_tensor] = 0.0
        self.successes[env_ids_tensor] = 0
        self.object_lifted[env_ids_tensor] = False
        
        # Write states
        self.robot.write_root_state_to_sim(root_state, env_ids_tensor)
        self.robot.write_joint_state_to_sim(all_joint_pos, joint_vel, None, env_ids_tensor)
        
        # Update buffers
        self.robot.update(dt=self.cfg.sim.dt)
        for obj in self.objects:
            obj.update(dt=self.cfg.sim.dt)
        
        print(f"[RESET] Hand positioned above target: {self._object_infos[0].object_id}")
        

   
@torch.jit.script
def quat_apply_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to vectors."""
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    
    t = 2 * torch.cross(xyz, vec, dim=-1)
    rotated = vec + w * t + torch.cross(xyz, t, dim=-1)
    
    return rotated


@torch.jit.script
def quat_apply_batch(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to batched vectors.
    
    Args:
        quat: (num_envs, 4) quaternions in (w, x, y, z) format
        vec: (num_envs, num_points, 3) vectors to rotate
    
    Returns:
        torch.Tensor: (num_envs, num_points, 3) rotated vectors
    """
    num_envs = quat.shape[0]
    num_points = vec.shape[1]
    
    # Expand quaternion to match point cloud shape
    quat_expanded = quat.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, 4)
    
    w = quat_expanded[..., 0:1]  # (B, N, 1)
    xyz = quat_expanded[..., 1:4]  # (B, N, 3)
    
    # Apply quaternion rotation
    t = 2 * torch.cross(xyz, vec, dim=-1)  # (B, N, 3)
    rotated = vec + w * t + torch.cross(xyz, t, dim=-1)  # (B, N, 3)
    
    return rotated


@torch.jit.script
def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (roll, pitch, yaw) to quaternions."""
    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    quat = torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)
    
    return quat


@torch.jit.script
def quaternion_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * np.pi / 2,
        torch.asin(sinp)
    )
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    return euler


@torch.jit.script
def wrap_angle_diff(angles: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences to [-Ï, Ï] range."""
    diff = angles - reference
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return torch.abs(diff)

@torch.jit.script
def wrap_angle_diff_for_limits(angles: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute SIGNED angle difference wrapped to [-Ï, Ï] range.
    
    Args:
        angles: Current angles (num_envs, 3)
        reference: Reference angles (num_envs, 3)
    
    Returns:
        Signed difference in [-Ï, Ï] range
    """
    diff = angles - reference
    # Wrap to [-Ï, Ï]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff

def compute_rewards_shaped(
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_init_pos: torch.Tensor,
    palm_euler: torch.Tensor,
    joint_pos: torch.Tensor,
    chamfer_distances: torch.Tensor,
    contact_forces: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    object_lifted: torch.Tensor,
    cfg,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    """Shaped reward function.
    
    Simple principle:
    1. Get hand close to object
    2. Close ALL fingers
    3. Lift the object
    
    """
    
    num_envs = hand_pos.shape[0]
    device = hand_pos.device
    
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    
    # Termination penalties
    table_surface_world = env_origins[:, 2] + cfg.table_height
    object_fell = object_pos[:, 2] < (table_surface_world - cfg.object_fall_margin)
    reward[object_fell] += -5.0
    
    object_pos_rel = object_pos - env_origins
    table_margin = 0.05
    table_x_limit = cfg.table_width / 2 - table_margin
    table_y_limit = cfg.table_depth / 2 - table_margin
    out_of_bounds = (torch.abs(object_pos_rel[:, 0]) > table_x_limit) | (torch.abs(object_pos_rel[:, 1]) > table_y_limit)
    reward[out_of_bounds] += -3.0
    
    # Height change
    dh = torch.clamp(object_pos[:, 2] - object_init_pos[:, 2], min=0.0)
    
    # Split chamfer distances
    finger_chamfer = chamfer_distances[:, :10]
    palm_chamfer = chamfer_distances[:, 10:]
    
    # Separate thumb (sites 0-1) from other fingers (sites 2-9)
    thumb_chamfer = finger_chamfer[:, 0:2]
    other_fingers_chamfer = finger_chamfer[:, 2:10]
    
    # Get minimum distances
    thumb_dist = torch.min(thumb_chamfer, dim=1)[0]
    other_fingers_dist = other_fingers_chamfer.mean(dim=1)  # Mean of 4 fingers
    palm_dist = palm_chamfer.mean(dim=1)  # Palm distance
    
    # Contact forces
    finger_forces = contact_forces[:, :10]
    palm_forces = contact_forces[:, 10:]
    
    thumb_forces = finger_forces[:, 0:2]
    other_finger_forces = finger_forces[:, 2:10]
    
    active_thumb = (thumb_forces > cfg.contact_force_threshold).any(dim=1)
    active_fingers = (other_finger_forces > cfg.contact_force_threshold).sum(dim=1)
    active_palm = (palm_forces > cfg.contact_force_threshold).any(dim=1)
    
    # Orientation and joint state
    ref_palm_euler = torch.tensor(cfg.ref_palm_euler, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
    ref_open_joints = torch.tensor(cfg.ref_open_joints, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
    
    angle_diff = wrap_angle_diff(palm_euler, ref_palm_euler)
    angle_diff_constrained = angle_diff.clone()
    angle_diff_constrained[:, 2] = 0.0

    state_diff = torch.abs(joint_pos - ref_open_joints)
    
    rot_penalty = (angle_diff_constrained / cfg.palm_orientation_tolerance).sum(dim=1)
    state_penalty = state_diff.mean(dim=1)
    
    # ============ SIMPLE 3-PHASE REWARD ============
    
    # Phase 1: Get hand close (already working from your first version)
    hand_to_object = torch.norm(hand_pos - object_pos, dim=1)
    reward_approach = cfg.alpha_hands * hand_to_object
    
    # Phase 2: Close ALL parts - thumb, fingers, AND palm must get close
    reward_thumb_close = cfg.alpha_thumb_close * thumb_dist
    reward_fingers_close = cfg.alpha_fingers_close * other_fingers_dist
    reward_palm_close = cfg.alpha_palm_close * palm_dist
    
    # Bonus when ALL parts are close (proper wrap/envelop grasp)
    thumb_is_close = thumb_dist < cfg.thumb_threshold
    fingers_are_close = other_fingers_dist < cfg.fingers_threshold
    palm_is_close = palm_dist < cfg.palm_threshold
    all_close = thumb_is_close & fingers_are_close & palm_is_close
    
    # Wrapping quality: Low variance = uniform contact (good wrap)
    # Include ALL finger sites (thumb + other fingers) for full hand wrapping
    # Only reward uniform wrapping when hand is in proper grasp position
    all_finger_variance = torch.var(finger_chamfer, dim=1)  # Variance across all 10 sites
    reward_wrap = torch.zeros(num_envs, device=device)
    reward_wrap[all_close] = cfg.alpha_wrap * all_finger_variance[all_close]

    reward_grasp_bonus = torch.zeros(num_envs, device=device)
    reward_grasp_bonus[all_close] = cfg.alpha_grasp
    
    # Phase 3: Lift object (only when properly grasping)
    can_lift = all_close | object_lifted
    lift_progress = dh / cfg.max_lift_height
    goal_height = cfg.max_lift_height
    goal_distance = torch.abs(dh - goal_height)
    # Reward hand moving upward when grasping
    hand_height_change = hand_pos[:, 2] - (object_init_pos[:, 2] + cfg.hand_init_height_above_object)
    hand_lift_progress = torch.clamp(hand_height_change / cfg.max_lift_height, 0.0, 1.0)

    reward_lift = torch.zeros(num_envs, device=device)
    reward_lift[can_lift] = cfg.alpha_lift * (lift_progress[can_lift] + 0.5 * hand_lift_progress[can_lift])
    

    
    # Small constraints (only when far from object)
    far_from_object = hand_to_object > 0.15
    reward_open = torch.zeros(num_envs, device=device)
    reward_open[far_from_object] = cfg.alpha_open * state_penalty[far_from_object]
    
    constrain_rotation = far_from_object
    reward_rot = torch.zeros(num_envs, device=device)
    reward_rot[constrain_rotation] = cfg.alpha_rot * rot_penalty[constrain_rotation]
    
    # Success bonus
    at_target_height = dh >= (cfg.max_lift_height * 0.6)
    success = all_close & at_target_height
    reward_success = torch.where(success, torch.tensor(cfg.alpha_success, device=device), torch.tensor(0.0, device=device))
    
    # Update lifted flag
    object_lifted = object_lifted | (all_close & (dh > cfg.min_lift_height))
    
    total_reward = (
        reward +
        reward_approach +           # Get close
        reward_thumb_close +        # Close thumb
        reward_fingers_close +      # Close other fingers
        reward_palm_close +         # Close palm
        reward_wrap +               # Good wraping
        reward_grasp_bonus +        # Bonus when all close
        reward_lift +               # Lift when grasping
        reward_success +            # Success bonus
        reward_open +               # Small constraint when far
        reward_rot                  # Small constraint when far
    )
    
    # Debug printing
    if cfg.enable_debug_reward and num_envs > 0:
        env_id = 0
        print("\n" + "="*60)
        print(f"[SIMPLE DEBUG] Environment {env_id}")
        print("="*60)
        print(f"Hand-to-object: {hand_to_object[env_id].item():.4f}m")
        print(f"Thumb distance: {thumb_dist[env_id].item():.4f}m (close: {thumb_is_close[env_id].item()})")
        print(f"Fingers distance: {other_fingers_dist[env_id].item():.4f}m (close: {fingers_are_close[env_id].item()})")
        print(f"Palm distance: {palm_dist[env_id].item():.4f}m (close: {palm_is_close[env_id].item()})")
        print(f"All close (grasp): {all_close[env_id].item()} {'â' if all_close[env_id].item() else 'â'}")
        print(f"Object lift: {dh[env_id].item():.4f}m")
        print(f"Goal distance: {goal_distance[env_id].item():.4f}m")
        print(f"\n[Rewards]")
        print(f"  approach: {reward_approach[env_id].item():.3f}")
        print(f"  thumb: {reward_thumb_close[env_id].item():.3f}")
        print(f"  fingers: {reward_fingers_close[env_id].item():.3f}")
        print(f"  palm: {reward_palm_close[env_id].item():.3f}")
        print(f"  grasp_bonus: {reward_grasp_bonus[env_id].item():.3f}")
        print(f"  lift: {reward_lift[env_id].item():.3f}")
        print(f"  success: {reward_success[env_id].item():.3f}")
        print(f"  open: {reward_open[env_id].item():.3f}")
        print(f"  rot: {reward_rot[env_id].item():.3f}")
        print(f"\nTOTAL: {total_reward[env_id].item():.3f}")
        print("="*60 + "\n")
    
    return total_reward