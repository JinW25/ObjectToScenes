# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Panda + parallel-gripper pick-and-lift benchmark, controlled with privileged-IK.

Compared to BenchmarkEnv (ContactileHand, PPO/clutter-RL/transformer, floating hand
root, chamfer-distance / point-cloud observations):

  * Robot:      fixed-base Franka Panda instead of a free-floating multi-fingered hand.
  * End-effector: 2-finger parallel (panda_hand + panda_finger_joint1/2) instead of the
                 ContactileHand's 6-actuated-joint multi-finger hand.
  * Controller: no learning. The object's root pose is read directly from the
                 simulator (privileged info) and the arm is driven to it with a
                 differential IK controller through a scripted state machine:
                     APPROACH -> DESCEND -> CLOSE_GRIPPER -> LIFT -> HOLD -> DONE
  * No point clouds, chamfer distances, contact-force shaping, or policy loading --
    none of that is needed when there's nothing to train.

What's reused from benchmark_env.py (robot-agnostic, copied/adapted):
  table creation, object USD spawning, clutter/random/isolated position generation,
  spawn validation, and the general DirectRLEnv scene-setup shape.
"""

from __future__ import annotations

import numpy as np
import time
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_mul, combine_frame_transforms
from pxr import Usd, UsdGeom
import omni.usd

from .panda_ik_benchmark_env_cfg import PandaIKBenchmarkEnvCfg, ObjectSpawnInfo, CLUTTER_CONFIGS
from .ggcnn_model import GGCNN, predict_grasp_candidates, predict_best_grasp, _gaussian_blur


# ── Phase constants for the scripted state machine ──────────────────────────
PHASE_OBSERVE = 0    # (retry only) drive back up to the vantage point, then recapture
PHASE_APPROACH = 1   # move above the CANDIDATE grasp (pregrasp hover), gripper open
PHASE_DESCEND = 2    # descend to the grasp height, gripper open
PHASE_CLOSE = 3       # close the gripper
PHASE_LIFT = 4        # lift to cfg.lift_height, gripper closed; verified the instant it's reached
PHASE_FREEZE = 5       # success confirmed -- hold perfectly still, then the trial ends immediately
PHASE_DONE = 6         # trial finished (success recorded); reset teleports the robot home
_PHASE_NAMES = ["OBSERVE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "FREEZE", "DONE"]


# ── Camera projection/unprojection, ported EXACTLY (same formulas, same
# quat_w_ros convention) from run_realworld_experiment.py's
# project_points_to_image_cached / unproject_pixel_to_world_with_depth /
# quat_to_rotation_matrix_ros -- not re-derived, since that reference is tested,
# working code and this project's own hand-rolled version (intrinsic_matrices +
# quat_apply, cx=width/2) had at least one real discrepancy against it
# (cx should be (width-1)/2, not width/2) plus an unverified quat_w_world
# attribute guess where the reference concretely uses quat_w_ros.

def _quat_to_rotation_matrix_ros(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(dtype=torch.float32)
    norm = torch.sqrt((quat**2).sum())
    w, x, y, z = quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], device=quat.device, dtype=torch.float32)


def _camera_intrinsics(focal_length: float, horizontal_aperture: float, width: int, height: int):
    f_x = (focal_length / horizontal_aperture) * width
    f_y = f_x
    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0
    return f_x, f_y, c_x, c_y


def _project_point_to_image_px(point_world, cam_pos_w, cam_quat_w, focal_length, horizontal_aperture, width, height):
    """Single-point version of project_points_to_image_cached. Returns (px, py) or
    None if the point is behind the camera.

    BUG FIX: this used rot_mat.T here, but the reference's actual formula for
    this direction (world -> camera) is `points_rel @ rot_mat` with NO
    transpose -- confirmed directly against project_points_to_image_cached,
    whose own `rot_mat.T.T` (double transpose = no-op) and inline comment
    ("project: point_cam = point_world_rel @ rot_mat") both say so explicitly.
    The transpose belongs on the OTHER direction (_unproject_pixel_to_world,
    which was already correct). Getting this backwards meant target_px -- the
    pixel a YOLO box gets matched against -- was computed with the rotation
    applied the wrong way, so the "nearest box to the target" search was
    matching against a scrambled reference point, not the target's real
    on-screen location.
    """
    cam_pos_w = cam_pos_w.to(dtype=torch.float32)
    rot_mat = _quat_to_rotation_matrix_ros(cam_quat_w)
    point_rel = point_world.to(dtype=torch.float32) - cam_pos_w
    point_cam = torch.matmul(point_rel, rot_mat)
    if float(point_cam[2]) <= 0.01:
        return None
    f_x, f_y, c_x, c_y = _camera_intrinsics(focal_length, horizontal_aperture, width, height)
    px = f_x * (float(point_cam[0]) / float(point_cam[2])) + c_x
    py = f_y * (float(point_cam[1]) / float(point_cam[2])) + c_y
    return px, py


def _unproject_pixel_to_world(px, py, depth, cam_pos_w, cam_quat_w, focal_length, horizontal_aperture, width, height, device):
    """Exact port of unproject_pixel_to_world_with_depth. distance_to_image_plane
    is the camera-frame Z coordinate (perpendicular distance, not ray length)."""
    f_x, f_y, c_x, c_y = _camera_intrinsics(focal_length, horizontal_aperture, width, height)
    rot_mat = _quat_to_rotation_matrix_ros(cam_quat_w)
    point_cam = torch.tensor(
        [depth * (px - c_x) / f_x, depth * (py - c_y) / f_y, depth], dtype=torch.float32, device=device
    )
    point_world_rel = torch.matmul(point_cam, rot_mat.T)
    return cam_pos_w.to(dtype=torch.float32) + point_world_rel


@torch.no_grad()
def _run_ggcnn_maps(model, depth_norm: torch.Tensor, device: str):
    """Run GG-CNN and return the full (H,W) quality/angle/width maps -- same
    preprocessing predict_best_grasp uses internally (sigmoid + gaussian blur
    for quality, atan2 for angle, sigmoid for width) but without its built-in
    argmax, so the caller can select by a different criterion (e.g. closest to
    a point among masked pixels, not just whichever pixel scored highest)."""
    x = depth_norm.to(device).float().unsqueeze(0).unsqueeze(0)
    pos_out, cos_out, sin_out, width_out = model(x)
    quality = torch.sigmoid(pos_out)
    quality = _gaussian_blur(quality)
    angle = torch.atan2(sin_out, cos_out) / 2.0
    width = torch.sigmoid(width_out)
    return quality[0, 0], angle[0, 0], width[0, 0]


class PandaIKBenchmarkEnv(DirectRLEnv):
    """Fixed-base Panda + parallel gripper, driven by privileged-info differential IK."""

    cfg: PandaIKBenchmarkEnvCfg

    def __init__(self, cfg: PandaIKBenchmarkEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg

        self.objects: list[RigidObject] = []
        self._object_infos: list[ObjectSpawnInfo] = []
        # Cache of each object USD's own local-frame axis-aligned bounding box
        # (min_xyz, max_xyz), keyed by usd_path -- computed once per distinct
        # object, reused every trial. See _get_object_local_aabb /
        # _project_object_privileged_box: this is what lets the target's image
        # bounding box come directly from privileged 3D geometry (like
        # multi_object_sequential_env.py's generate_individual_object_images
        # projects each object's own known mesh points) instead of depending on
        # SAM correctly guessing which class-agnostic mask is the target.
        self._object_local_aabb_cache: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {}
        self._current_trial = 0
        self._trial_results: list[dict] = []
        # Wall-clock start time of the CURRENT trial (one per env), matching
        # BenchmarkEnv's picking_time = time.time() - self._trial_start_time.
        self._trial_start_time: list[float | None] = [None] * cfg.scene.num_envs

        self._load_available_objects()
        self._select_objects_to_spawn()

        # Robot base must be placed before the Articulation is spawned in _setup_scene(),
        # which happens inside super().__init__(). table_width isn't known until now,
        # so this can't live as a static default in the cfg dataclass.
        # Floating just OUTSIDE the table edge (not mounted on the tabletop itself) --
        # at z = table_height, x pushed past the edge by robot_edge_gap. This avoids
        # two problems an on-table mount had: (1) the base's own collision geometry
        # interpenetrating the tabletop cuboid when the inset got small, which fights
        # the IK-driven motion with contact forces and can make the arm never actually
        # reach anything; (2) objects spawned right next to the base sitting in its
        # unreachable near-base dead zone (joint limits / self-collision close in).
        robot_x = -(self.cfg.table_width / 2.0 + self.cfg.robot_edge_gap)
        robot_z = self.cfg.table_height
        self.cfg.robot_cfg.init_state.pos = (robot_x, 0.0, robot_z)

        super().__init__(cfg, render_mode, **kwargs)

        # ── Joint / body indices ────────────────────────────────────────────
        self._arm_joint_ids, self._arm_joint_names = self.robot.find_joints("panda_joint.*")
        self._gripper_joint_ids, _ = self.robot.find_joints(list(self.cfg.gripper_joint_names))
        ee_body_ids, _ = self.robot.find_bodies(self.cfg.ee_body_name)
        self._ee_body_idx = ee_body_ids[0]
        # Isaac Lab's per-body jacobian tensor excludes the fixed base link, so body
        # index N in robot.data corresponds to row N-1 in the jacobian tensor.
        self._ee_jacobi_idx = self._ee_body_idx - 1

        self._default_arm_joint_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

        # ── IK controller ────────────────────────────────────────────────────
        self.ik_controller = DifferentialIKController(
            self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        # Fixed top-down grasp orientation (panda_hand frame). This is the standard
        # "fingers pointing straight down" quaternion (w, x, y, z) used in the Isaac
        # Lab Franka lift examples -- VERIFY against your actual panda_hand asset by
        # printing self.robot.data.body_quat_w[:, self._ee_body_idx] at the default
        # joint pose before trusting this in a real run; different USD variants of
        # the Franka hand can have a different neutral frame.
        self._down_quat_w = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1).clone()

        # ── State machine buffers ───────────────────────────────────────────
        self._phase = torch.full((self.num_envs,), PHASE_APPROACH, dtype=torch.long, device=self.device)
        self._phase_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gripper_cmd = torch.full(
            (self.num_envs,), self.cfg.gripper_open_pos, dtype=torch.float32, device=self.device
        )
        self._target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_b = self._down_quat_w.clone()
        self._object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._trial_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._regrasp_attempts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Frozen the instant CLOSE finishes (i.e. before any lifting has happened) --
        # the LIFT/FREEZE target is computed from THIS, not from the object's live
        # position, which rises every step once it's actually being carried upward.
        self._lift_grasp_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._lift_grasp_quat_w = self._down_quat_w.clone()
        # GG-CNN-predicted (or naive-fallback) grasp pose, computed once per trial
        # (see _capture_and_compute_grasp) and used to drive APPROACH/DESCEND/CLOSE
        # for that whole trial -- not recomputed live, same "freeze it" reasoning
        # as the LIFT-target fix above.
        self._grasp_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._grasp_quat_w = self._down_quat_w.clone()
        self._grasp_width_m = torch.full(
            (self.num_envs,), self.cfg.gripper_open_pos * 2, dtype=torch.float32, device=self.device
        )
        # Wall-clock seconds spent capturing/processing during the CURRENT trial
        # (initial capture + any retry recaptures) -- subtracted from picking_time
        # so it only reflects active movement. Reset to 0 each trial.
        self._trial_paused_s = torch.zeros(self.num_envs, device=self.device)
        # Set True when a misgrasp retry sends the arm back up through APPROACH;
        # the recapture actually happens once it GETS there (approach_done), not
        # at the moment the retry is decided (arm is still down near the table then).
        self._pending_recapture = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Set True by _capture_and_compute_grasp whenever it had to resort to
        # the naive object-root grasp for that env, False whenever it found a
        # genuine vision-derived point. Checked by callers (_reset_idx, the
        # retry handler) to route a naive result into ANOTHER retry instead of
        # ever accepting it outright -- see close_gripper_fully-adjacent
        # "never fall back to naive" handling.
        self._used_naive_fallback = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── GG-CNN grasp predictor (optional -- empty checkpoint path disables it
        # and every trial falls back to the naive object-root grasp) ────────────
        self._ggcnn_model = None
        if self.cfg.ggcnn_checkpoint:
            try:
                self._ggcnn_model = GGCNN().to(self.cfg.ggcnn_device)
                state_dict = torch.load(self.cfg.ggcnn_checkpoint, map_location=self.cfg.ggcnn_device)
                self._ggcnn_model.load_state_dict(state_dict, strict=True)
                self._ggcnn_model.eval()
                print(f"[INFO] GG-CNN loaded from {self.cfg.ggcnn_checkpoint}")
            except Exception as exc:
                print(f"[WARN] Failed to load GG-CNN checkpoint ({exc}); "
                      f"falling back to naive object-root grasping for every trial.")
                self._ggcnn_model = None
        else:
            print("[INFO] No ggcnn_checkpoint set -- using naive object-root grasping.")

        # ── SAM object localizer (optional -- empty checkpoint path disables it
        # and every capture falls back to whole-frame GG-CNN candidate matching).
        # SAM is class-agnostic segmentation, not detection -- it proposes masks
        # for "things" without naming them, which is why matching (below) is
        # ALWAYS geometric (privileged target position projected into the image),
        # unlike the class-name matching a per-object-trained YOLO could do. ────
        self._sam_mask_generator = None
        if self.cfg.sam_checkpoint:
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                print(f"[SAM] Loading {self.cfg.sam_model_type} from: {self.cfg.sam_checkpoint}")
                sam = sam_model_registry[self.cfg.sam_model_type](checkpoint=self.cfg.sam_checkpoint)
                sam.to(device=self.cfg.ggcnn_device)
                self._sam_mask_generator = SamAutomaticMaskGenerator(
                    sam, points_per_side=self.cfg.sam_points_per_side
                )
                print(f"[SAM] Loaded")
            except Exception as exc:
                print(f"[WARN] Failed to load SAM ({exc}); falling back to whole-frame "
                      f"GG-CNN candidate matching (no crop).")
                self._sam_mask_generator = None
        else:
            print("[INFO] No sam_checkpoint set -- using whole-frame GG-CNN candidate matching (no crop).")

        print(f"[INFO] Panda IK benchmark ready. Target object: {self._object_infos[0].object_id}")
        self._initialize_scene_objects()

        # One-time calibration: solve for the arm joint configuration that puts the
        # end effector cfg.observe_height above the table center, ONCE, here at
        # construction. Every _reset_idx afterward writes straight to this cached
        # config in a single instant joint-state write -- no per-trial convergence
        # motion. (The table center is a fixed point relative to the robot base, so
        # this only needs solving once, not every trial.)
        self._start_arm_joint_pos = self._default_arm_joint_pos.clone()
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._move_ee_to_start_pose(all_env_ids)
        self._start_arm_joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids].clone()

    # ─────────────────────────────────────────────────────────────────────
    # Object catalogue / selection (adapted from BenchmarkEnv, no policies)
    # ─────────────────────────────────────────────────────────────────────

    def _load_available_objects(self):
        usd_dir = Path(self.cfg.object_usd_dir)
        if not usd_dir.exists():
            raise FileNotFoundError(f"Object USD directory not found: {usd_dir}")
        for usd_file in sorted(usd_dir.glob("*.usd")):
            self.cfg.available_objects.append(
                ObjectSpawnInfo(object_id=usd_file.stem, usd_path=str(usd_file))
            )
        if len(self.cfg.available_objects) == 0:
            raise RuntimeError(f"No USD files found in {usd_dir}")
        print(f"[INFO] Loaded {len(self.cfg.available_objects)} candidate objects from {usd_dir}")

    def _select_objects_to_spawn(self):
        """Pick target object (index 0, the one the arm will pick) + optional clutter."""
        if self.cfg.target_object_id:
            target = next((o for o in self.cfg.available_objects if o.object_id == self.cfg.target_object_id), None)
            if target is None:
                raise ValueError(f"Target object '{self.cfg.target_object_id}' not found")
        else:
            target = self.cfg.available_objects[np.random.randint(len(self.cfg.available_objects))]

        self._object_infos = [target]

        if self.cfg.use_isolated_mode:
            print(f"[SPAWN] Isolated mode -- target only: {target.object_id}")
            return

        if self.cfg.use_clutter_based_spawn:
            config = CLUTTER_CONFIGS[self.cfg.target_complexity]
            num_neighbors = np.random.randint(config['num_neighbors'][0], config['num_neighbors'][1] + 1)
        else:
            num_total = np.random.randint(self.cfg.min_objects_to_spawn, self.cfg.max_objects_to_spawn + 1)
            num_neighbors = max(0, num_total - 1)

        pool = [o for o in self.cfg.available_objects if o.object_id != target.object_id]
        if num_neighbors > 0 and pool:
            idxs = np.random.choice(len(pool), size=num_neighbors, replace=True)
            self._object_infos.extend(pool[i] for i in idxs)

        print(f"[SPAWN] Target: {target.object_id} + {len(self._object_infos) - 1} clutter object(s) "
              f"(clutter is passive -- only the target at index 0 is picked)")

    # ─────────────────────────────────────────────────────────────────────
    # Scene setup
    # ─────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        for i, obj_info in enumerate(self._object_infos):
            object_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Object_{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=obj_info.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=False, disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, self.cfg.table_height + 0.05),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            obj = RigidObject(object_cfg)
            self.objects.append(obj)
            self.scene.rigid_objects[f"object_{i}"] = obj

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self._create_table_in_source()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        self.grasp_camera = Camera(self.cfg.grasp_camera)
        self.scene.sensors["grasp_camera"] = self.grasp_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print(f"[INFO] Scene setup complete: Panda + {len(self.objects)} object(s)")

    def _create_table_in_source(self):
        """Identical to BenchmarkEnv._create_table_in_source (robot-agnostic)."""
        leg_height = self.cfg.table_height - self.cfg.table_thickness
        source_env_path = "/World/envs/env_0"

        table_top_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.table_width, self.cfg.table_depth, self.cfg.table_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        table_top_cfg.func(
            f"{source_env_path}/Table/TableTop", table_top_cfg,
            translation=(0.0, 0.0, self.cfg.table_height - self.cfg.table_thickness / 2),
        )

        leg_offset_x = self.cfg.table_width / 2 - self.cfg.leg_radius - 0.02
        leg_offset_y = self.cfg.table_depth / 2 - self.cfg.leg_radius - 0.02
        leg_positions = [
            (leg_offset_x, leg_offset_y, leg_height / 2), (-leg_offset_x, leg_offset_y, leg_height / 2),
            (leg_offset_x, -leg_offset_y, leg_height / 2), (-leg_offset_x, -leg_offset_y, leg_height / 2),
        ]
        for i, pos in enumerate(leg_positions):
            leg_cfg = sim_utils.CylinderCfg(
                radius=self.cfg.leg_radius, height=leg_height,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            leg_cfg.func(f"{source_env_path}/Table/Leg{i}", leg_cfg, translation=pos)

    # ─────────────────────────────────────────────────────────────────────
    # Object placement (adapted from BenchmarkEnv's position generators)
    # ─────────────────────────────────────────────────────────────────────

    def _initialize_scene_objects(self, env_ids: Sequence[int] | None = None):
        """Place all objects on the table and let physics settle. Robot-agnostic."""
        num_objects = len(self.objects)
        env_origins = self.scene.env_origins

        for attempt in range(self.cfg.max_spawn_attempts):
            if self.cfg.use_isolated_mode:
                positions = self._generate_isolated_position()
            elif self.cfg.use_clutter_based_spawn:
                positions = self._generate_clutter_based_positions(num_objects)
            else:
                positions = self._generate_random_positions(num_objects)

            for i, obj in enumerate(self.objects):
                pos_rel = positions[i]
                obj_pos_world = env_origins.clone()
                obj_pos_world[:, 0] += pos_rel[0]
                obj_pos_world[:, 1] += pos_rel[1]
                obj_pos_world[:, 2] += self.cfg.table_height + 0.05

                if self.cfg.randomize_object_orientation:
                    if self.cfg.randomize_object_yaw_only:
                        yaw = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * np.pi
                        half = yaw / 2
                        quat = torch.stack(
                            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
                            dim=-1,
                        )
                    else:
                        rpy = (torch.rand(self.num_envs, 3, device=self.device) * 2 - 1) * np.pi
                        quat = _euler_to_quat(rpy)
                else:
                    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)

                state = obj.data.default_root_state.clone()
                state[:, 0:3] = obj_pos_world
                state[:, 3:7] = quat
                state[:, 7:] = 0.0
                obj.write_root_state_to_sim(state)

            for _ in range(self.cfg.spawn_settling_steps):
                self.sim.step(render=False)
                for obj in self.objects:
                    obj.update(dt=self.cfg.sim.dt)

            if self._validate_spawn_configuration():
                break
            elif attempt == self.cfg.max_spawn_attempts - 1:
                print("[WARN] Failed to reach a valid spawn configuration after max attempts, proceeding anyway")

        target = self.objects[0]
        target.update(dt=self.cfg.sim.dt)
        self._object_init_pos = target.data.root_pos_w.clone()

    def _generate_isolated_position(self) -> list:
        # Target always at table center, regardless of spawn mode.
        return [(0.0, 0.0)]

    def _generate_clutter_based_positions(self, num_objects: int) -> list:
        config = CLUTTER_CONFIGS[self.cfg.target_complexity]
        positions = [(0.0, 0.0)]  # target always at table center; neighbors surround it
        num_neighbors = num_objects - 1
        margin = self.cfg.spawn_area_margin
        table_x_limit = self.cfg.table_width / 2 - margin
        table_y_limit = self.cfg.table_depth / 2 - margin

        for i in range(num_neighbors):
            base_angle = (i / max(num_neighbors, 1)) * 2 * np.pi
            angle = base_angle + np.random.uniform(-0.3, 0.3)
            distance = np.random.uniform(config['min_clearance'], config['max_clearance'])
            x = np.clip(distance * np.cos(angle), -table_x_limit, table_x_limit)
            y = np.clip(distance * np.sin(angle), -table_y_limit, table_y_limit)
            positions.append((x, y))
        return positions

    def _generate_random_positions(self, num_objects: int) -> list:
        # Target (index 0) always at table center, regardless of spawn mode --
        # only clutter objects (index 1+, if any) get randomized positions,
        # placed around the target rather than anywhere on the table.
        positions = [(0.0, 0.0)]
        margin = self.cfg.spawn_area_margin
        usable_width = self.cfg.table_width - 2 * margin
        usable_depth = self.cfg.table_depth - 2 * margin

        for i in range(1, num_objects):
            placed = False
            for _ in range(100):
                x = (torch.rand(1).item() - 0.5) * usable_width
                y = (torch.rand(1).item() - 0.5) * usable_depth
                if min(np.hypot(x - px, y - py) for px, py in positions) >= self.cfg.min_object_spacing:
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                positions.append(self._get_grid_position(i, num_objects, usable_width, usable_depth))
        return positions

    @staticmethod
    def _get_grid_position(index: int, total: int, width: float, depth: float) -> tuple:
        cols = int(np.ceil(np.sqrt(total)))
        rows = int(np.ceil(total / cols))
        col, row = index % cols, index // cols
        x = (col + 1) * (width / (cols + 1)) - width / 2
        y = (row + 1) * (depth / (rows + 1)) - depth / 2
        return (x, y)

    def _validate_spawn_configuration(self) -> bool:
        env_origins = self.scene.env_origins
        for obj in self.objects:
            pos = obj.data.root_pos_w[0]
            height_above_table = pos[2] - (env_origins[0, 2] + self.cfg.table_height)
            if height_above_table < -self.cfg.spawn_height_tolerance or height_above_table > 0.1:
                return False
            dist_from_center = torch.norm(pos[:2] - env_origins[0, :2]).item()
            if dist_from_center > self.cfg.max_spawn_distance_from_origin:
                return False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Scripted IK state machine
    # ─────────────────────────────────────────────────────────────────────

    def _tcp_target_to_hand_target(self, tcp_pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        """Offset a desired fingertip (TCP) position back to the panda_hand frame origin."""
        offset = torch.tensor([0.0, 0.0, self.cfg.tcp_offset_z], device=self.device).expand(self.num_envs, -1)
        return tcp_pos_w - quat_apply(quat_w, offset)

    def _get_ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def _move_ee_to_start_pose(self, env_ids: torch.Tensor) -> None:
        """Drive the arm to a fixed home pose directly above the table center, at
        the start of each trial, before the normal APPROACH phase takes over.

        Uses the exact same mechanics as regular stepping -- compute the Jacobian,
        solve one IK step, write the joint target, then advance the sim with a real
        `sim.step()` -- just called directly in a loop rather than through
        `env.step()`. This is the same proven pattern already used to let spawned
        objects settle (see `_initialize_scene_objects`), unlike an instantaneous
        joint-state write with no physics in between, which does not reliably
        converge and previously destabilized the arm.

        Unlike every other phase transition in the state machine (APPROACH/
        DESCEND/OBSERVE-retry all gate on `pos_err < pose_reach_pos_tol`, with a
        timeout as a fallback -- see `_pre_physics_step`), this ONE-TIME
        calibration used to just run a fixed `observe_settle_steps` iterations
        and cache whatever pose it happened to land on, with no check that it
        actually got there. Since this result is cached once into
        `_start_arm_joint_pos` and instant-written on every single reset for
        the rest of the run, an unconverged calibration here silently wrecks
        EVERY trial's camera geometry identically and invisibly -- exactly the
        "target_px far from image center, every capture" symptom. Now it exits
        as soon as it actually converges (same `pose_reach_pos_tol` the rest of
        the state machine trusts), and loudly reports if it hits the iteration
        cap without converging instead of caching a silently-bad pose.
        """
        env_origins = self.scene.env_origins
        target_pos_w = env_origins.clone()
        target_pos_w[:, 2] += self.cfg.table_height + self.cfg.observe_height
        hand_target_pos_w = self._tcp_target_to_hand_target(target_pos_w, self._down_quat_w)
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, hand_target_pos_w, self._down_quat_w
        )

        self.ik_controller.reset(env_ids)
        self.ik_controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))

        # Generous cap, NOT the number of steps this is expected to need --
        # observe_settle_steps (100) measured 18.5% of the required lateral
        # travel with no sign of having plateaued, so 100 is just not enough
        # real sim time (0.83s at dt=1/120) for a PD-controlled arm to traverse
        # ~0.73m from its default joint pose. This loop exits early via the
        # convergence check below on any run where it converges well before
        # the cap, so raising this costs nothing in the common case.
        max_steps = max(self.cfg.observe_settle_steps, 500)
        pos_err = None
        converged_at = None
        for step_i in range(max_steps):
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._arm_joint_ids]
            joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
            ee_pos_b, ee_quat_b = self._get_ee_pose_b()
            joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            all_joint_pos_target = self.robot.data.joint_pos_target.clone()
            all_joint_pos_target[:, self._arm_joint_ids] = joint_pos_des
            all_joint_pos_target[:, self._gripper_joint_ids] = self.cfg.gripper_open_pos
            self.robot.set_joint_position_target(all_joint_pos_target)
            self.robot.write_data_to_sim()

            self.sim.step(render=False)
            self.robot.update(dt=self.cfg.sim.dt)

            pos_err = torch.norm(ee_pos_b - target_pos_b, dim=-1)
            if step_i % 50 == 0 or step_i == max_steps - 1:
                print(f"[CALIB] observe-pose step {step_i}: pos_err={float(pos_err.max()):.4f}m "
                      f"(target <{self.cfg.pose_reach_pos_tol:.4f}m)")
            if bool((pos_err < self.cfg.pose_reach_pos_tol).all()):
                converged_at = step_i
                break

        if converged_at is not None:
            print(f"[CALIB] Observe-pose calibration converged after {converged_at + 1} step(s), "
                  f"final pos_err={float(pos_err.max()):.4f}m.")
        else:
            ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
            print(f"[CALIB] WARNING: observe-pose calibration did NOT converge within {max_steps} steps "
                  f"(final pos_err={float(pos_err.max()):.4f}m, tol={self.cfg.pose_reach_pos_tol:.4f}m). "
                  f"Every trial's wrist camera will be calibrated to this WRONG pose -- "
                  f"target hand_target_pos_w={hand_target_pos_w[0].tolist()}, "
                  f"actual ee_pos_w={ee_pos_w[0].tolist()}. Raise max_steps above, check the IK "
                  f"controller gains/lambda_val, or verify the target is actually within the arm's "
                  f"reach before trusting anything downstream of this calibration.")

        # ── DIAGNOSTIC (temporary) ──────────────────────────────────────────
        # Separates two very different bugs that both show up downstream as
        # "target_px far from image center every capture":
        #   (a) the ARM isn't actually reaching the intended world pose despite
        #       pos_err reporting convergence (a root_pos_w/root_quat_w or
        #       reach-limit problem), vs.
        #   (b) the arm gets there fine, but the wrist camera's local mount
        #       offset/rotation (grasp_camera.offset in the cfg) is wrong.
        # Compare the three world positions/quats printed below against the
        # intended target -- hand_target_pos_w (already computed above) is
        # what panda_hand should have converged to. If ee_pos_w is close to
        # hand_target_pos_w but grasp_camera's pos_w is NOT close to ee_pos_w,
        # it's (b) -- fix grasp_camera.offset in panda_ik_benchmark_env_cfg.py.
        # If ee_pos_w itself is far from hand_target_pos_w, it's (a) -- check
        # root_pos_w/root_quat_w below against the assumed base pose
        # (table_width/2 + robot_edge_gap, 0, table_height) with IDENTITY
        # rotation; a non-identity root_quat_w would silently invalidate the
        # subtract_frame_transforms math this whole state machine relies on.
        ee_pos_w_diag = self.robot.data.body_pos_w[:, self._ee_body_idx]
        root_pos_w_diag = self.robot.data.root_pos_w
        root_quat_w_diag = self.robot.data.root_quat_w
        print(f"[DIAG] hand_target_pos_w (intended) = {hand_target_pos_w[0].tolist()}")
        print(f"[DIAG] panda_hand ee_pos_w (actual)  = {ee_pos_w_diag[0].tolist()}")
        print(f"[DIAG] robot root_pos_w              = {root_pos_w_diag[0].tolist()}")
        print(f"[DIAG] robot root_quat_w             = {root_quat_w_diag[0].tolist()} "
              f"(expected identity-ish (1,0,0,0) if the base has no baked-in rotation)")
        try:
            self.grasp_camera.update(dt=self.cfg.sim.dt, force_recompute=True)
            cam_pos_w_diag = self.grasp_camera.data.pos_w
            cam_quat_w_diag = self.grasp_camera.data.quat_w_ros
            print(f"[DIAG] grasp_camera pos_w             = {cam_pos_w_diag[0].tolist()}")
            print(f"[DIAG] grasp_camera quat_w_ros         = {cam_quat_w_diag[0].tolist()}")
            print(f"[DIAG] ee_pos_w -> grasp_camera pos_w delta = "
                  f"{(cam_pos_w_diag[0] - ee_pos_w_diag[0]).tolist()} "
                  f"(should be small, ~grasp_camera.offset.pos magnitude, e.g. <0.1m)")
        except Exception as exc:
            print(f"[DIAG] could not read grasp_camera pose yet: {exc}")
        try:
            analytic_pos, analytic_quat = self._compute_wrist_camera_world_pose()
            print(f"[DIAG] ANALYTIC camera pos_w (from ee pose + local mount, "
                  f"bypassing the sensor entirely) = {analytic_pos[0].tolist()} "
                  f"(should now be close to ee_pos_w, ~0.06m apart)")
            print(f"[DIAG] ANALYTIC camera quat_w_ros = {analytic_quat[0].tolist()}")
        except Exception as exc:
            print(f"[DIAG] analytic camera pose computation failed: {exc}")

        # ── DIAGNOSTIC PART 2: inspect the actual USD prim hierarchy ────────
        # The 0.43m gap above is way bigger than the configured 0.06m offset
        # could ever produce (rotating a 0.06m vector can change its direction
        # but never its length) -- so this isn't a rotation-sign bug, it means
        # the camera prim genuinely isn't sitting where grasp_camera.prim_path
        # says it should. Check the real stage directly: does a "panda_hand"
        # prim exist where expected, is WristCamera actually its CHILD, and
        # what LOCAL transform does USD have recorded for it (should be
        # translate=(0,0,0.06) if the offset was applied as intended)?
        try:
            stage = omni.usd.get_context().get_stage()
            hand_path = "/World/envs/env_0/Robot/panda_hand"
            cam_path = hand_path + "/WristCamera"
            hand_prim = stage.GetPrimAtPath(hand_path)
            cam_prim = stage.GetPrimAtPath(cam_path)
            print(f"[DIAG] USD prim at {hand_path} valid: {hand_prim.IsValid()}")
            print(f"[DIAG] USD prim at {cam_path} valid: {cam_prim.IsValid()}")
            if cam_prim.IsValid():
                actual_parent = cam_prim.GetParent().GetPath()
                print(f"[DIAG] WristCamera's ACTUAL USD parent: {actual_parent} "
                      f"(expected: {hand_path})")
                local_xform = UsdGeom.Xformable(cam_prim).GetLocalTransformation()
                local_translate = local_xform.ExtractTranslation()
                print(f"[DIAG] WristCamera LOCAL translation (relative to its actual "
                      f"parent): {tuple(local_translate)} (expected ~(0, 0, 0.06))")

                # Local attachment confirmed above -- now ask the STAGE directly,
                # right now, what the camera's live WORLD transform actually is,
                # completely bypassing TiledCamera's own Python-side pos_w/quat_w_ros
                # cache. If this matches panda_hand's current ee_pos_w (it should,
                # given the local offset is only 0.06m), but grasp_camera.data.pos_w
                # above does NOT, that proves the TiledCamera sensor object's cached
                # world pose is stale/decoupled from the actual simulated transform
                # hierarchy -- i.e. the bug is in how/when the sensor refreshes its
                # own pose bookkeeping, not in the offset config or the USD parenting.
                world_xform = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                world_translate = world_xform.ExtractTranslation()
                print(f"[DIAG] WristCamera LIVE USD world translation (stage query, "
                      f"bypasses TiledCamera's cached pos_w): {tuple(world_translate)}")
                print(f"[DIAG]   compare to ee_pos_w={ee_pos_w_diag[0].tolist()} (should be close, "
                      f"~0.06m apart) and to grasp_camera.data.pos_w reported above "
                      f"(if THAT one disagrees with this live stage query, the TiledCamera "
                      f"sensor's cached pose is the bug, not the mount/offset config).")
            else:
                # Try to find it anywhere under Robot, in case it landed at an
                # unexpected path (e.g. wrong link name in the wildcard match).
                robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
                found = [str(p.GetPath()) for p in Usd.PrimRange(robot_prim) if "WristCamera" in p.GetName()]
                print(f"[DIAG] No prim at expected cam_path. Any 'WristCamera' prim "
                      f"anywhere under /World/envs/env_0/Robot? {found or 'NONE FOUND'}")
                # Also list the direct children of Robot so we can see the real
                # link names if 'panda_hand' itself doesn't exist as expected.
                if not hand_prim.IsValid():
                    all_links = [str(p.GetPath()) for p in Usd.PrimRange(robot_prim)]
                    print(f"[DIAG] 'panda_hand' prim not found. All prims under Robot "
                          f"(check the real hand/wrist link name here): {all_links}")
        except Exception as exc:
            print(f"[DIAG] USD hierarchy inspection failed: {exc}")
        # ── END DIAGNOSTIC ───────────────────────────────────────────────────

        # Leave the controller's internal state clean -- the upcoming APPROACH
        # phase sets its own command on the very next _pre_physics_step.
        self.ik_controller.reset(env_ids)

    def _grasp_height_is_plausible(self, point_w: torch.Tensor, env_idx: int) -> bool:
        """Sanity check on an unprojected grasp position: anywhere on the table,
        an object's actual surface should sit within a modest band above the
        tabletop, not near the camera or floating implausibly high. Catches a
        wrong-region match (wrong YOLO box, wrong whole-frame candidate) or a
        camera-pose/projection bug producing a technically-computed-but-
        physically-nonsensical point, BEFORE sending the arm to chase it, rather
        than silently trusting whatever number came out of the unprojection.
        """
        table_top_z = self.scene.env_origins[env_idx, 2] + self.cfg.table_height
        z = float(point_w[2])
        lo, hi = float(table_top_z) - 0.05, float(table_top_z) + 0.30
        if lo <= z <= hi:
            return True
        print(f"[GGCNN] env {env_idx}: REJECTING candidate -- unprojected z={z:.3f} is outside "
              f"the plausible table-surface band [{lo:.3f}, {hi:.3f}] (table_top={float(table_top_z):.3f}). "
              f"This points at a wrong region match or a camera-pose/projection bug, not a real "
              f"object surface -- falling back rather than sending the arm to chase it.")
        return False

    def _get_wrist_camera_local_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read WristCamera's LOCAL translate + rotation relative to its true
        parent (panda_hand) directly from USD, once, and cache it. This is the
        one piece of the mount geometry we've directly verified is correct
        (translate == the configured (0,0,0.06); parent == panda_hand) -- see
        the [DIAG] USD prim inspection this was built to confirm. Cached as
        (local_pos (3,), local_quat_wxyz (4,)) tensors on self.device.
        """
        if getattr(self, "_wrist_cam_local_pose_cache", None) is not None:
            return self._wrist_cam_local_pose_cache
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_hand/WristCamera")
        if not cam_prim.IsValid():
            raise RuntimeError(
                "Cannot compute wrist camera pose analytically: no prim at "
                "/World/envs/env_0/Robot/panda_hand/WristCamera. Check grasp_camera.prim_path."
            )
        local_xform = UsdGeom.Xformable(cam_prim).GetLocalTransformation()
        t = local_xform.ExtractTranslation()
        q = local_xform.ExtractRotationQuat()  # Gf.Quatd: real + imaginary Vec3d
        local_pos = torch.tensor([t[0], t[1], t[2]], dtype=torch.float32, device=self.device)
        imag = q.GetImaginary()
        local_quat = torch.tensor(
            [q.GetReal(), imag[0], imag[1], imag[2]], dtype=torch.float32, device=self.device
        )
        self._wrist_cam_local_pose_cache = (local_pos, local_quat)
        print(f"[DIAG] cached WristCamera local pose from USD: pos={local_pos.tolist()}, "
              f"quat(wxyz)={local_quat.tolist()}")
        return self._wrist_cam_local_pose_cache

    def _compute_wrist_camera_world_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytically compute the wrist camera's TRUE world pos/quat (ROS
        optical convention, matching what _project_point_to_image_px /
        _unproject_pixel_to_world expect) from panda_hand's live tensor-API
        pose -- NOT from any camera-sensor-reported pos_w/quat_w_ros, which we
        confirmed (via two sensor classes AND a raw USD stage query all
        agreeing on the same wrong, frozen value) cannot be trusted for this
        wrist-mounted setup in this pipeline.

        Composition: world_pose(camera) = world_pose(panda_hand) o local_pose
        (USD/world convention), then apply the standard fixed OpenGL/USD ->
        ROS optical-frame correction (180 deg about the resulting frame's own
        local X axis) since the downstream projection math is written against
        quat_w_ros.

        Returns (cam_pos_w (num_envs,3), cam_quat_w_ros (num_envs,4)).
        """
        local_pos, local_quat = self._get_wrist_camera_local_pose()
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        num_envs = ee_pos_w.shape[0]
        local_pos_b = local_pos.unsqueeze(0).expand(num_envs, -1)
        local_quat_b = local_quat.unsqueeze(0).expand(num_envs, -1)
        cam_pos_w, cam_quat_w_usd = combine_frame_transforms(ee_pos_w, ee_quat_w, local_pos_b, local_quat_b)
        ros_fix = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        ros_fix_b = ros_fix.unsqueeze(0).expand(num_envs, -1)
        cam_quat_w_ros = quat_mul(cam_quat_w_usd, ros_fix_b)
        return cam_pos_w, cam_quat_w_ros

    def _capture_and_compute_grasp(self, env_ids: torch.Tensor) -> float:
        """Capture the top-down camera, run GG-CNN over the WHOLE scene, and pick
        whichever candidate grasp lands closest to the target object's own
        (privileged) coordinate -- that's what enforces "only grasp the target,"
        not any segmentation/masking. Falls back to the naive object-root grasp
        if GG-CNN is disabled, nothing usable comes back from the camera, or the
        best candidate is too far from the target to plausibly BE the target.

        Called once per trial before any movement, and again on each misgrasp
        retry (from up at hover height -- see the RELOCATE handling in
        _pre_physics_step). Caller is expected to add the returned duration to
        the trial's paused time so picking_time only counts active movement.
        """
        t0 = time.time()
        target_obj = self.objects[0]
        fallback_pos_w = target_obj.data.root_pos_w.clone()
        fallback_quat_w = self._down_quat_w.clone()
        # BUG FIX: this used to be gripper_open_pos * 2, which made close_target
        # (= width/2, clamped to [closed,open]) evaluate to EXACTLY gripper_open_pos
        # -- i.e. "closed" == "open", so the gripper never actually closed on any
        # fallback grasp. Naive/fallback grasping has no predicted object width, so
        # it should close all the way, same as the pre-GG-CNN behavior.
        fallback_width_m = torch.full_like(self._grasp_width_m, self.cfg.gripper_closed_pos * 2)

        if self._ggcnn_model is None:
            self._grasp_pos_w[env_ids] = fallback_pos_w[env_ids]
            self._grasp_quat_w[env_ids] = fallback_quat_w[env_ids]
            self._grasp_width_m[env_ids] = fallback_width_m[env_ids]
            self._used_naive_fallback[env_ids] = True
            return time.time() - t0

        # Advance the sim one render step so the camera buffer actually reflects
        # the current scene (freshly spawned objects, or -- on a retry -- the arm
        # having just moved up out of the way).
        self.scene.update(dt=self.cfg.sim.dt)
        self.sim.render()
        self.grasp_camera.update(dt=self.cfg.sim.dt, force_recompute=True)

        cam_data = self.grasp_camera.data
        if "distance_to_image_plane" not in cam_data.output:
            print(f"[WARN] grasp_camera missing distance_to_image_plane output "
                  f"({list(cam_data.output.keys())}); falling back to naive grasp. "
                  f"Did you launch with --enable_cameras?")
            self._grasp_pos_w[env_ids] = fallback_pos_w[env_ids]
            self._grasp_quat_w[env_ids] = fallback_quat_w[env_ids]
            self._grasp_width_m[env_ids] = fallback_width_m[env_ids]
            self._used_naive_fallback[env_ids] = True
            return time.time() - t0
        cam_cfg = self.cfg.grasp_camera
        focal_length = cam_cfg.spawn.focal_length
        aperture = cam_cfg.spawn.horizontal_aperture
        img_width, img_height = cam_cfg.width, cam_cfg.height

        # Analytically-derived camera world pose (see _compute_wrist_camera_world_pose)
        # -- used INSTEAD of cam_data.pos_w / cam_data.quat_w_ros, which we confirmed
        # (across TiledCamera, Camera, and a raw USD stage query) are all stuck
        # reporting a stale, incorrect pose for this wrist-mounted setup. Only
        # cam_data.output[...] (the actual rendered rgb/depth) is still read from
        # the sensor -- that part looks geometrically sane on inspection.
        cam_pos_w_all, cam_quat_w_all = self._compute_wrist_camera_world_pose()
        # Gates every debug-image WRITE in this function (console logging is
        # unaffected) -- see ggcnn_debug_save_trial_stride's cfg comment for why
        # this matters at batch-experiment scale.
        stride = max(1, self.cfg.ggcnn_debug_save_trial_stride)
        debug_save_this_capture = self.cfg.ggcnn_debug_save_images and (self._current_trial % stride == 0)

        for i in env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids):
            depth = cam_data.output["distance_to_image_plane"][i, ..., 0]  # (H, W)

            # One-time raw scene snapshot -- ONLY the very first trial of the
            # whole run (_current_trial==1, 1-indexed), and only its first
            # capture (regrasp_attempts still 0, i.e. not a retry). Saves the
            # camera's rgb output undecorated, plus the raw depth array
            # losslessly (.npy), independent of ggcnn_debug_save_trial_stride.
            if self.cfg.save_scene_images and self._current_trial == 1 and int(self._regrasp_attempts[i].item()) == 0:
                self._save_scene_snapshot(cam_data, depth, i)

            # ── Sanity-check the raw capture BEFORE trusting anything downstream ──
            depth_np = depth.detach().cpu().numpy()
            d_min, d_max, d_mean, d_std = float(depth_np.min()), float(depth_np.max()), float(depth_np.mean()), float(depth_np.std())
            near, far = self.cfg.grasp_camera.spawn.clipping_range
            frac_at_near = float((depth_np <= near * 1.01).mean())
            frac_at_far = float((depth_np >= far * 0.99).mean())
            print(f"[GGCNN] env {i} raw depth stats: min={d_min:.4f} max={d_max:.4f} mean={d_mean:.4f} "
                  f"std={d_std:.4f} (clip range {near:.3f}-{far:.3f}, "
                  f"{frac_at_near*100:.1f}% at near limit, {frac_at_far*100:.1f}% at far limit)")

            degenerate = d_std < self.cfg.ggcnn_debug_min_depth_std or (frac_at_near + frac_at_far) > 0.95
            if degenerate:
                print(f"[WARN] env {i}: grasp_camera depth looks degenerate (flat/at-clip-limit) -- "
                      f"almost certainly a mounting/orientation problem (camera facing into its own "
                      f"mount, wrong clipping_range for the actual distance, or --enable_cameras not "
                      f"actually rendering this camera), NOT a GG-CNN problem. Treating this as a failed "
                      f"attempt to retry rather than trusting a prediction built on garbage input.")
                self._used_naive_fallback[i] = True
                self._grasp_pos_w[i] = fallback_pos_w[i]
                self._grasp_quat_w[i] = fallback_quat_w[i]
                self._grasp_width_m[i] = fallback_width_m[i]
                if debug_save_this_capture:
                    self._save_debug_grasp_image(depth_np, [], None, i, tag="DEGENERATE")
                continue

            cam_pos_w = cam_pos_w_all[i]
            # Analytically computed (see _compute_wrist_camera_world_pose) --
            # NOT read from cam_data.quat_w_ros, which we confirmed is stale.
            cam_quat_w = cam_quat_w_all[i]
            target_pos_w_i = target_obj.data.root_pos_w[i]
            target_px = _project_point_to_image_px(
                target_pos_w_i, cam_pos_w, cam_quat_w, focal_length, aperture, img_width, img_height
            )

            # Geometry sanity check: the target is ALWAYS at table center (see
            # _generate_isolated/random/clutter_positions), and the observation
            # pose is ALSO always directly above table center looking straight
            # down -- so if the camera pose/orientation is right, target_px
            # should land almost exactly at the image center, every single
            # capture, regardless of clutter. If it doesn't, that's direct
            # evidence of a camera mounting/orientation problem, independent of
            # anything about YOLO or matching.
            img_cx, img_cy = (img_width - 1) / 2.0, (img_height - 1) / 2.0
            print(f"[GEOM] env {i}: cam_pos_w={cam_pos_w.tolist()} cam_quat_w(ros)={cam_quat_w.tolist()}")
            if target_px is not None:
                off_x, off_y = target_px[0] - img_cx, target_px[1] - img_cy
                flag = "" if (abs(off_x) < img_width * 0.05 and abs(off_y) < img_height * 0.05) else \
                    "  <-- FAR FROM CENTER: camera orientation is almost certainly wrong"
                print(f"[GEOM] env {i}: target_px={target_px}, image_center=({img_cx:.1f},{img_cy:.1f}), "
                      f"offset=({off_x:+.1f},{off_y:+.1f}){flag}")

            # ── The target's box is now computed DIRECTLY from privileged 3D
            # geometry (its own USD's local AABB, transformed by its live pose,
            # projected corner-by-corner) -- not by asking SAM to guess which
            # class-agnostic mask is the target. This is the same trick
            # multi_object_sequential_env.py's generate_individual_object_images
            # uses (project each object's own known mesh points), and it sidesteps
            # the whole "which SAM box is right" ambiguity entirely: given a
            # correct camera pose, this box IS the target, by construction, no
            # matching/guessing involved. SAM still runs (if enabled) purely to
            # draw the other proposals for visual context in the debug image --
            # it no longer decides identity. ────────────────────────────────────
            target_box_px = self._project_object_privileged_box(
                0, i, cam_pos_w, cam_quat_w, focal_length, aperture, img_width, img_height
            )
            if target_box_px is not None:
                print(f"[BOX] env {i}: target's privileged bounding box = {target_box_px} "
                      f"(projected from its own USD AABB + live pose -- not a SAM guess)")
            else:
                print(f"[BOX] env {i}: target's AABB projected entirely behind/at the camera -- "
                      f"no box available this capture.")

            detections = []
            if self._sam_mask_generator is not None:
                rgb_out = cam_data.output.get("rgb") if hasattr(cam_data.output, "get") else (
                    cam_data.output["rgb"] if "rgb" in cam_data.output else None
                )
                if rgb_out is None:
                    print(f"[SAM] env {i}: grasp_camera has no rgb output -- add 'rgb' to "
                          f"grasp_camera.data_types. Skipping SAM proposals for this capture.")
                else:
                    rgb_np = rgb_out[i, ..., :3].detach().cpu().numpy()
                    if rgb_np.dtype != np.uint8:
                        rgb_np = np.clip(rgb_np * 255.0 if rgb_np.max() <= 1.0 else rgb_np, 0, 255).astype(np.uint8)
                    detections = self._detect_sam_boxes(rgb_np)
                    if debug_save_this_capture:
                        self._save_sam_debug_image(rgb_np, detections, target_box_px, target_px, i)

            matched_box = target_box_px

            # ── GG-CNN always runs on the WHOLE frame, never a crop -- cropping
            # was previously used to isolate the target before inference, but
            # that meant a wrong/bad box didn't just mis-rank candidates, it
            # threw every other region away before GG-CNN even saw them. That's
            # still true here: the network sees the whole scene. What changed
            # is SELECTION -- instead of taking GG-CNN's global top-K quality
            # peaks and hoping enough of them happen to land on the target
            # (which fails outright in clutter if the target isn't among the
            # highest-quality-LOOKING objects), predict_best_grasp's valid_mask
            # restricts the ARGMAX itself to the target's privileged box, so
            # the result is guaranteed to be the best point genuinely on the
            # target, not a hopeful match off a fixed-size candidate list. ────
            size = self.cfg.ggcnn_input_size
            depth_resized = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0), size=(size, size), mode="nearest"
            )[0, 0]
            depth_norm = torch.clamp(depth_resized - depth_resized.mean(), -1.0, 1.0)

            scale_r = depth.shape[0] / size
            scale_c = depth.shape[1] / size

            # Debug-only: whole-frame candidate peaks, purely to draw the cyan
            # "other candidates" dots in the saved debug image for visual
            # context -- NOT used for selection anymore (see below).
            candidates = predict_grasp_candidates(
                self._ggcnn_model, depth_norm, device=self.cfg.ggcnn_device, top_k=self.cfg.ggcnn_top_k
            )
            candidates = [c for c in candidates if c["quality"] >= self.cfg.ggcnn_min_quality]
            for c in candidates:
                c["orig_row"] = c["row"] * scale_r
                c["orig_col"] = c["col"] * scale_c

            best = None
            selection_tag = None
            if matched_box is not None:
                m = self.cfg.sam_crop_margin_px
                bx0, by0, bx1, by1 = matched_box[0] - m, matched_box[1] - m, matched_box[2] + m, matched_box[3] + m
                # matched_box is in NATIVE (img_width x img_height) pixels, but
                # depth_norm -- what the model actually sees -- is resized to
                # ggcnn_input_size. Scale the box down by the same factor used
                # everywhere else to convert native pixels to network pixels.
                # This rectangle is only a SEARCH WINDOW (bounds where we bother
                # checking at all) -- NOT the mask itself, see below.
                net_x0 = max(0, int(bx0 / scale_c))
                net_x1 = min(size - 1, int(bx1 / scale_c))
                net_y0 = max(0, int(by0 / scale_r))
                net_y1 = min(size - 1, int(by1 / scale_r))

                # FIX: a rectangular mask can't tell the target apart from a
                # neighboring object whose own pixels also happen to fall
                # inside this (already-loosened, see rotation-AABB discussion)
                # box -- confirmed happening in practice with tightly-packed
                # clutter (argmax picked the neighbor's higher-quality point
                # since it was technically "in the box"). Instead, for every
                # pixel in the search window, unproject using ITS OWN observed
                # depth and keep it only if that real 3D point actually lands
                # near the target's true (privileged) position -- a
                # neighboring object's pixels unproject to a clearly different
                # XY location even when their boxes overlap, so this tells
                # them apart where a rectangle alone can't.
                valid_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)
                if net_x1 >= net_x0 and net_y1 >= net_y0:
                    footprint = self._get_object_world_footprint_extent(0, i)
                    accept_radius = (
                        max(footprint) / 2.0 * self.cfg.grasp_width_margin if footprint is not None else 0.04
                    )
                    rows_idx = torch.arange(net_y0, net_y1 + 1, device=self.device)
                    cols_idx = torch.arange(net_x0, net_x1 + 1, device=self.device)
                    rr, cc = torch.meshgrid(rows_idx, cols_idx, indexing="ij")
                    orig_rr = rr.float() * scale_r
                    orig_cc = cc.float() * scale_c
                    depths_window = depth_resized[rr, cc]
                    f_x_win, f_y_win, c_x_win, c_y_win = _camera_intrinsics(focal_length, aperture, img_width, img_height)
                    rot_mat_win = _quat_to_rotation_matrix_ros(cam_quat_w)
                    x_cam = depths_window * (orig_cc - c_x_win) / f_x_win
                    y_cam = depths_window * (orig_rr - c_y_win) / f_y_win
                    points_cam = torch.stack([x_cam, y_cam, depths_window], dim=-1)  # (Hs,Ws,3)
                    points_world = torch.matmul(points_cam, rot_mat_win.T) + cam_pos_w
                    xy_dist = torch.norm(points_world[..., :2] - target_pos_w_i[:2], dim=-1)
                    valid_mask[net_y0:net_y1 + 1, net_x0:net_x1 + 1] = xy_dist <= accept_radius
                    print(f"[GGCNN] env {i}: proximity mask accept_radius={accept_radius:.4f}m, "
                          f"{int(valid_mask.sum())}/{valid_mask.numel()} pixels kept in search window")

                quality_2d, angle_2d, width_2d = _run_ggcnn_maps(self._ggcnn_model, depth_norm, self.cfg.ggcnn_device)
                candidate_mask = valid_mask & (quality_2d >= self.cfg.ggcnn_min_quality)
                if candidate_mask.any():
                    # Among pixels that both pass the depth-verified proximity
                    # mask AND clear min_quality, pick the one closest to the
                    # box's own center rather than the single highest-quality
                    # pixel -- a center-ish grasp point is generally more
                    # stable than an edge point that happened to score higher,
                    # and this was requested after seeing GG-CNN sometimes pick
                    # an off-center point when a more central one was right there.
                    rows_c, cols_c = torch.where(candidate_mask)
                    center_row_net = (net_y0 + net_y1) / 2.0
                    center_col_net = (net_x0 + net_x1) / 2.0
                    dists_to_center = (rows_c.float() - center_row_net) ** 2 + (cols_c.float() - center_col_net) ** 2
                    sel = int(torch.argmin(dists_to_center))
                    row_sel, col_sel = int(rows_c[sel]), int(cols_c[sel])
                    result = {
                        "row": row_sel, "col": col_sel,
                        "angle_rad": float(angle_2d[row_sel, col_sel]),
                        "width": float(width_2d[row_sel, col_sel]),
                        "quality": float(quality_2d[row_sel, col_sel]),
                    }
                else:
                    result = {"row": None, "col": None, "angle_rad": 0.0, "width": 0.5, "quality": 0.0}
                quality_2d, angle_2d, width_2d = _run_ggcnn_maps(self._ggcnn_model, depth_norm, self.cfg.ggcnn_device)
                candidate_mask = valid_mask & (quality_2d >= self.cfg.ggcnn_min_quality)
                relaxed_quality = False
                if not candidate_mask.any():
                    # Tier 2, before ever considering naive: relax the quality
                    # gate but KEEP the depth-verified proximity mask -- any
                    # point genuinely on the target beats giving up just
                    # because everything there happened to score below
                    # min_quality. Still strictly on-target, never off-object.
                    candidate_mask = valid_mask & (quality_2d > 0.0)
                    relaxed_quality = True
                if candidate_mask.any():
                    # Among surviving pixels, pick the one closest to the
                    # box's own center rather than the single highest-quality
                    # pixel -- a center-ish grasp point is generally more
                    # stable than an edge point that happened to score higher,
                    # and this was requested after seeing GG-CNN sometimes pick
                    # an off-center point when a more central one was right there.
                    rows_c, cols_c = torch.where(candidate_mask)
                    center_row_net = (net_y0 + net_y1) / 2.0
                    center_col_net = (net_x0 + net_x1) / 2.0
                    dists_to_center = (rows_c.float() - center_row_net) ** 2 + (cols_c.float() - center_col_net) ** 2
                    sel = int(torch.argmin(dists_to_center))
                    row_sel, col_sel = int(rows_c[sel]), int(cols_c[sel])
                    result = {
                        "row": row_sel, "col": col_sel,
                        "angle_rad": float(angle_2d[row_sel, col_sel]),
                        "width": float(width_2d[row_sel, col_sel]),
                        "quality": float(quality_2d[row_sel, col_sel]),
                    }
                else:
                    result = {"row": None, "col": None, "angle_rad": 0.0, "width": 0.5, "quality": 0.0}
                if result["row"] is not None:
                    orig_row = result["row"] * scale_r
                    orig_col = result["col"] * scale_c
                    pixel_depth = float(depth[int(orig_row), int(orig_col)])
                    point_w = _unproject_pixel_to_world(
                        orig_col, orig_row, pixel_depth, cam_pos_w, cam_quat_w,
                        focal_length, aperture, img_width, img_height, self.device
                    )
                    if self._grasp_height_is_plausible(point_w, i):
                        best = dict(result)
                        best["orig_row"], best["orig_col"] = orig_row, orig_col
                        best["pixel_depth"], best["point_w"] = pixel_depth, point_w
                        selection_tag = "IN_TARGET_MASK" if not relaxed_quality else "IN_TARGET_MASK_RELAXED"
                        note = (f" [WARNING: quality gate relaxed -- nothing in the mask cleared "
                                f"min_quality={self.cfg.ggcnn_min_quality}, this point is below that bar]"
                                if relaxed_quality else "")
                        print(f"[GGCNN] env {i}: chose MASKED candidate at pixel "
                              f"({int(orig_row)},{int(orig_col)}) quality={result['quality']:.3f} -- "
                              f"closest to the target box's center among all pixels that passed the "
                              f"depth-verified proximity mask {matched_box} (+-{m}px search window){note}")
                    else:
                        print(f"[GGCNN] env {i}: masked argmax landed at an implausible height -- discarding")
                else:
                    print(f"[GGCNN] env {i}: proximity mask is completely empty (zero pixels with any "
                          f"positive quality landed near the target's true position) -- target may be "
                          f"fully occluded or the mask window degenerate this capture")
            else:
                print(f"[GGCNN] env {i}: no privileged box available -- cannot restrict search to the target")

            if best is None:
                self._used_naive_fallback[i] = True
                print(f"[GGCNN] env {i}: WARNING -- no usable vision-derived point found even after "
                      f"relaxing the quality gate; this attempt will be treated as a failure and retried "
                      f"(see close_gripper_fully-adjacent 'never fall back to naive' handling in the "
                      f"caller) rather than silently accepting a naive grasp.")
                self._grasp_pos_w[i] = fallback_pos_w[i]
                self._grasp_quat_w[i] = fallback_quat_w[i]
                self._grasp_width_m[i] = fallback_width_m[i]
                if debug_save_this_capture:
                    self._save_debug_grasp_image(depth_np, candidates, None, i, tag="NO_MATCH", scale=(scale_r, scale_c))
                continue

            grasp, point_w, pixel_depth = best, best["point_w"], best["pixel_depth"]
            print(f"[GGCNN] env {i}: ({selection_tag}) angle={np.degrees(grasp['angle_rad']):.1f}deg "
                  f"world_pos={point_w.tolist()}")

            half = grasp["angle_rad"] / 2.0
            yaw_quat = torch.tensor(
                [np.cos(half), 0.0, 0.0, np.sin(half)], device=self.device, dtype=torch.float32
            )
            grasp_quat_w = quat_mul(yaw_quat.unsqueeze(0), self._down_quat_w[i:i+1])[0]

            f_x, _, _, _ = _camera_intrinsics(focal_length, aperture, img_width, img_height)
            # GG-CNN's width channel doesn't transfer across cameras the way
            # position/angle roughly do: it encodes an absolute meters-per-pixel
            # relationship the network only ever saw for the Cornell dataset's
            # own camera (fixed focal length + typical depth). Applying that to
            # a completely different camera/depth setup without retraining
            # produces a systematically wrong magnitude, not noise -- confirmed
            # empirically (raw predictions of ~0.19-0.20m vs a max possible
            # gripper opening of 0.08m, consistently, across different grasps
            # on the same object). Recalibrating ggcnn_width_px_scale by trial
            # and error chases a moving target with no ground truth to check
            # against.
            #
            # Since this whole benchmark already privileges target identification
            # and its bounding box from the object's own known 3D geometry (see
            # _project_object_privileged_box), do the same for width: measure
            # the target's TRUE world-space footprint directly instead of
            # trusting the foreign-camera-trained width channel for an absolute
            # distance. GG-CNN still supplies POSITION and ANGLE (those are
            # scale-invariant / geometric, and transfer far better).
            width_px_native = grasp["width"] * self.cfg.ggcnn_width_px_scale * scale_c
            width_m_ggcnn_raw = width_px_native * pixel_depth / f_x  # logged only, not used

            if self.cfg.close_gripper_fully:
                width_m = self.cfg.gripper_closed_pos * 2
                print(f"[GGCNN] env {i}: close_gripper_fully=True -- commanding full close "
                      f"({width_m:.4f}m) rather than estimating an object-specific width "
                      f"[GG-CNN's own raw width prediction was {width_m_ggcnn_raw:.4f}m, for reference only]")
            else:
                footprint = self._get_object_world_footprint_extent(0, i)
                if footprint is not None:
                    extent_x, extent_y = footprint
                    # The gripper's closing direction only needs to clear the
                    # object's NARROWER horizontal dimension -- a good grasp
                    # angle naturally aligns the jaws with that shorter axis
                    # anyway, so this is a reasonable proxy regardless of the
                    # exact predicted angle, without needing to project the AABB
                    # onto that specific angle.
                    true_width_m = min(extent_x, extent_y) * self.cfg.grasp_width_margin
                    width_m = float(np.clip(true_width_m, self.cfg.gripper_closed_pos * 2, self.cfg.gripper_open_pos * 2))
                    print(f"[GGCNN] env {i}: width from privileged footprint={true_width_m:.4f}m "
                          f"(object extent_x={extent_x:.4f}m, extent_y={extent_y:.4f}m, "
                          f"margin={self.cfg.grasp_width_margin}) -> clamped={width_m:.4f}m "
                          f"[GG-CNN's own raw width prediction was {width_m_ggcnn_raw:.4f}m, for comparison only]")
                else:
                    # No cached AABB for this object's USD (shouldn't normally
                    # happen) -- fall back to the naive fixed default rather than
                    # trusting the uncalibrated GG-CNN magnitude.
                    width_m = self.cfg.gripper_open_pos  # conservative mid-range default
                    print(f"[GGCNN] env {i}: WARNING no privileged footprint available for width -- "
                          f"using default {width_m:.4f}m. GG-CNN's own raw prediction was "
                          f"{width_m_ggcnn_raw:.4f}m (not used).")

            self._used_naive_fallback[i] = False
            self._grasp_pos_w[i] = point_w
            self._grasp_quat_w[i] = grasp_quat_w
            self._grasp_width_m[i] = width_m
            print(f"[GGCNN] env {i}: final gripper width={width_m:.4f}m "
                  f"(closed_pos={self.cfg.gripper_closed_pos:.3f}, open_pos={self.cfg.gripper_open_pos:.3f} "
                  f"per finger -- close_target = clamp(width/2, closed, open))")
            if debug_save_this_capture:
                width_px_for_display = width_m * f_x / pixel_depth
                self._save_debug_grasp_image(depth_np, candidates, grasp, i, tag=selection_tag,
                                              scale=(scale_r, scale_c), width_px=width_px_for_display)

        return time.time() - t0

    def _detect_sam_boxes(self, image_np) -> list:
        """Ported from the visual_grasp_classifier.py reference's
        run_sam_detection: SAM's automatic mask generator proposes masks for
        everything it can find (including background/table), so two area
        filters (sam_min_area_frac, sam_max_area_frac) keep only plausible
        individual objects -- same filtering, same reasoning as that script.
        Returns a list of (x0,y0,x1,y1) pixel boxes, largest-first. Class-
        agnostic: no name/identity is ever produced here, only "something is
        here" -- see _match_box_containing_or_nearest for how identity gets
        resolved (always geometrically, via the target's privileged position).
        """
        if self._sam_mask_generator is None:
            return []
        try:
            masks = self._sam_mask_generator.generate(image_np)
            img_h, img_w = image_np.shape[:2]
            img_area = img_h * img_w
            candidates = [
                m for m in masks
                if self.cfg.sam_min_area_frac <= (m["area"] / img_area) <= self.cfg.sam_max_area_frac
            ]
            candidates.sort(key=lambda m: m["area"], reverse=True)
            boxes = []
            for m in candidates:
                x, y, w, h = m["bbox"]  # SAM returns XYWH
                boxes.append((int(x), int(y), int(x + w), int(y + h)))
            print(f"[SAM]   {len(boxes)} proposal(s) survived area filtering "
                  f"(of {len(masks)} raw masks)")
            return boxes
        except Exception as e:
            print(f"[SAM] WARNING: detection failed ({e}); falling back to whole-frame matching")
        return []

    def _get_object_world_footprint_extent(self, obj_idx: int, env_idx: int) -> tuple[float, float] | None:
        """The target's true horizontal (world XY) footprint extent, in
        meters, from its own local AABB corners transformed by its live pose
        -- same privileged-geometry recipe as _project_object_privileged_box,
        just measuring world-space extent instead of projecting to pixels.
        Returns (extent_x, extent_y) or None if no AABB is cached for this
        object's USD yet.
        """
        aabb = self._get_object_local_aabb(self._object_infos[obj_idx].usd_path)
        if aabb is None:
            return None
        bmin, bmax = aabb
        obj = self.objects[obj_idx]
        root_pos_w = obj.data.root_pos_w[env_idx]
        root_quat_w = obj.data.root_quat_w[env_idx:env_idx + 1]
        corners_local = torch.stack([
            torch.tensor([x, y, z], dtype=torch.float32, device=self.device)
            for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])
        ])
        corners_world = quat_apply(root_quat_w.expand(8, -1), corners_local) + root_pos_w
        extent_x = float(corners_world[:, 0].max() - corners_world[:, 0].min())
        extent_y = float(corners_world[:, 1].max() - corners_world[:, 1].min())
        return extent_x, extent_y

    def _get_object_local_aabb(self, usd_path: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Local-frame axis-aligned bounding box of an object's own USD asset,
        computed once per distinct usd_path and cached (same recipe as
        multi_object_sequential_env.py's _verify_mesh_extraction: open the
        USD standalone, UsdGeom.BBoxCache on its default prim). This is a
        static property of the mesh -- independent of where any spawned
        instance currently sits in the world -- so it's safe to compute once
        and reuse across every trial that spawns this same object.
        """
        if usd_path in self._object_local_aabb_cache:
            return self._object_local_aabb_cache[usd_path]
        result = None
        try:
            stage = Usd.Stage.Open(usd_path)
            root_prim = stage.GetDefaultPrim() if stage else None
            if root_prim:
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
                bbox_range = bbox_cache.ComputeWorldBound(root_prim).ComputeAlignedRange()
                bmin, bmax = bbox_range.GetMin(), bbox_range.GetMax()
                result = (
                    torch.tensor([bmin[0], bmin[1], bmin[2]], dtype=torch.float32, device=self.device),
                    torch.tensor([bmax[0], bmax[1], bmax[2]], dtype=torch.float32, device=self.device),
                )
            else:
                print(f"[WARN] Could not open USD or find default prim for AABB: {usd_path}")
        except Exception as e:
            print(f"[WARN] Failed to compute local AABB for {usd_path}: {e}")
        self._object_local_aabb_cache[usd_path] = result
        return result

    def _project_object_privileged_box(
        self, obj_idx: int, env_idx: int, cam_pos_w, cam_quat_w, focal_length, aperture, img_width, img_height,
    ) -> tuple | None:
        """The target's image bounding box, computed directly from privileged
        3D geometry instead of asking SAM to guess which class-agnostic mask
        is the target: take the object's own local AABB corners (8 of them),
        transform to world using its ACTUAL live pose, project each corner
        through the (now camera-geometry-verified-correct) same
        _project_point_to_image_px used for the [GEOM] single-point check, and
        take the pixel-space min/max. Same idea as
        multi_object_sequential_env.py's generate_individual_object_images,
        which projects each object's own known mesh points to get its box --
        just using 8 AABB corners here instead of a full sampled mesh, which
        is enough for a bounding box and needs no extra mesh-sampling utility.

        Returns None only if every corner projects behind the camera (should
        never happen for an object sitting on the table in front of a
        downward-looking overhead camera).
        """
        aabb = self._get_object_local_aabb(self._object_infos[obj_idx].usd_path)
        if aabb is None:
            return None
        bmin, bmax = aabb
        obj = self.objects[obj_idx]
        root_pos_w = obj.data.root_pos_w[env_idx]
        root_quat_w = obj.data.root_quat_w[env_idx:env_idx + 1]
        corners_local = torch.stack([
            torch.tensor([x, y, z], dtype=torch.float32, device=self.device)
            for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])
        ])
        corners_world = quat_apply(root_quat_w.expand(8, -1), corners_local) + root_pos_w

        xs, ys = [], []
        for corner in corners_world:
            px = _project_point_to_image_px(
                corner, cam_pos_w, cam_quat_w, focal_length, aperture, img_width, img_height
            )
            if px is not None:
                xs.append(px[0])
                ys.append(px[1])
        if not xs:
            return None
        return (
            max(0, int(min(xs))), max(0, int(min(ys))),
            min(img_width - 1, int(max(xs))), min(img_height - 1, int(max(ys))),
        )

    def _match_box_containing_or_nearest(self, boxes: list, target_px: tuple) -> tuple | None:
        """UNUSED as of the privileged-geometry box change -- identity now comes
        directly from _project_object_privileged_box (the target's own USD AABB
        projected via its live pose), not from matching target_px against SAM's
        class-agnostic proposals. Left defined in case that's wanted again
        (e.g. as a cross-check against the privileged box); nothing calls this.

        Auto-selects the target's box using ONLY the privileged position (SAM
        has no object identity to match by, unlike a per-object-trained YOLO).
        Primary criterion: does target_px fall INSIDE this box -- if several
        contain it (overlapping proposals, common in dense clutter), the
        SMALLEST one wins, since a big box containing the point is more likely
        a loose/background proposal than a tight fit on the actual object.
        Falls back to nearest-centroid only if no box contains the point at all
        (e.g. it landed just outside a tightly-cropped mask edge).
        """
        if not boxes:
            return None
        tx, ty = target_px

        containing = [b for b in boxes if b[0] <= tx <= b[2] and b[1] <= ty <= b[3]]
        if containing:
            best = min(containing, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            return (max(0, int(best[0])), max(0, int(best[1])),
                    min(self.cfg.grasp_camera.width - 1, int(best[2])),
                    min(self.cfg.grasp_camera.height - 1, int(best[3])))

        best_box, best_dist = None, None
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            dist = (cx - tx) ** 2 + (cy - ty) ** 2
            if best_dist is None or dist < best_dist:
                best_dist, best_box = dist, b
        if best_box is None:
            return None
        # Unlike the whole-frame GG-CNN fallback (which is gated by
        # ggcnn_max_candidate_dist), this had no distance sanity check at all --
        # it would confidently return an arbitrarily-far, wrong box rather than
        # admit "nothing nearby matched." Cap it the same way.
        if best_dist > self.cfg.sam_nearest_fallback_max_px ** 2:
            return None
        return (max(0, int(best_box[0])), max(0, int(best_box[1])),
                min(self.cfg.grasp_camera.width - 1, int(best_box[2])),
                min(self.cfg.grasp_camera.height - 1, int(best_box[3])))

    def _ggcnn_on_crop(
        self, crop: torch.Tensor, crop_offset: tuple, full_depth: torch.Tensor,
        cam_pos_w, cam_quat_w, focal_length, aperture, img_width, img_height, env_idx: int,
    ):
        """UNUSED as of the no-crop pipeline change -- _capture_and_compute_grasp
        now always runs GG-CNN on the whole frame and filters candidates by
        whether they land inside the matched SAM box, instead of cropping to
        that box before inference. Left defined (not deleted) in case a crop-
        based path is wanted again later; nothing currently calls this.

        Run GG-CNN on an already-target-isolated crop and take the single best
        candidate (no privileged-position tie-break needed here -- YOLO already
        resolved identity by matching the box to the target before this crop was
        even made; this step only answers WHERE/HOW to grasp within it, not
        WHICH object it is). Returns (pos_w, quat_w, width_m) or None if nothing
        in the crop clears ggcnn_min_quality.
        """
        x0, y0 = crop_offset
        crop_h, crop_w = crop.shape[-2], crop.shape[-1]
        size = self.cfg.ggcnn_input_size
        crop_resized = F.interpolate(crop.unsqueeze(0).unsqueeze(0), size=(size, size), mode="nearest")[0, 0]
        crop_norm = torch.clamp(crop_resized - crop_resized.mean(), -1.0, 1.0)

        candidates = predict_grasp_candidates(
            self._ggcnn_model, crop_norm, device=self.cfg.ggcnn_device, top_k=self.cfg.ggcnn_top_k
        )
        candidates = [c for c in candidates if c["quality"] >= self.cfg.ggcnn_min_quality]
        if not candidates:
            if self.cfg.ggcnn_debug_save_images:
                # BUG FIX: this used to save `crop` (native crop resolution) with
                # no scale correction, while candidate row/col are in the RESIZED
                # (ggcnn_input_size) coordinate space -- markers landed in the
                # wrong place, or off-canvas entirely, on anything but a
                # coincidentally-300x300 crop. Saving crop_resized instead means
                # this is pixel-for-pixel what GG-CNN actually saw, so row/col
                # need no rescaling to draw correctly.
                crop_np = crop_resized.detach().cpu().numpy()
                self._save_debug_grasp_image(crop_np, [], None, env_idx, tag="CROP_NO_MATCH")
            return None

        best = candidates[0]  # already quality-sorted by predict_grasp_candidates
        scale_r, scale_c = crop_h / size, crop_w / size
        crop_row, crop_col = best["row"] * scale_r, best["col"] * scale_c
        full_row, full_col = y0 + crop_row, x0 + crop_col
        pixel_depth = float(full_depth[int(full_row), int(full_col)])

        point_w = _unproject_pixel_to_world(
            full_col, full_row, pixel_depth, cam_pos_w, cam_quat_w,
            focal_length, aperture, img_width, img_height, self.device
        )
        if not self._grasp_height_is_plausible(point_w, env_idx):
            if self.cfg.ggcnn_debug_save_images:
                crop_np = crop_resized.detach().cpu().numpy()
                self._save_debug_grasp_image(crop_np, candidates, best, env_idx, tag="CROP_BAD_HEIGHT")
            return None

        half = best["angle_rad"] / 2.0
        yaw_quat = torch.tensor([np.cos(half), 0.0, 0.0, np.sin(half)], device=self.device, dtype=torch.float32)
        grasp_quat_w = quat_mul(yaw_quat.unsqueeze(0), self._down_quat_w[env_idx:env_idx + 1])[0]

        f_x, _, _, _ = _camera_intrinsics(focal_length, aperture, img_width, img_height)
        width_px = best["width"] * self.cfg.ggcnn_width_px_scale
        width_m = width_px * pixel_depth / f_x
        width_m = float(np.clip(width_m, self.cfg.gripper_closed_pos * 2, self.cfg.gripper_open_pos * 2))

        print(f"[GGCNN] env {env_idx}: chose crop candidate at full-frame pixel "
              f"({int(full_row)},{int(full_col)}) quality={best['quality']:.3f} "
              f"angle={np.degrees(best['angle_rad']):.1f}deg world_pos={point_w.tolist()}")

        if self.cfg.ggcnn_debug_save_images:
            # Same fix as CROP_NO_MATCH above -- crop_resized (the actual network
            # input) instead of crop (native resolution), so the marker aligns
            # with what GG-CNN actually saw.
            crop_np = crop_resized.detach().cpu().numpy()
            # width_px above is in crop-NATIVE pixels (it's derived via the
            # full-frame focal length, and cropping alone doesn't change pixel
            # scale) but this image is crop_resized (network-input resolution) --
            # convert native->resized the same way row/col already do (dividing
            # by scale_r/scale_c instead of multiplying). The rectangle is
            # rotated by angle_rad, and scale_r/scale_c can differ on a
            # non-square crop, so this averages them rather than handling the
            # (rarer) anisotropic case exactly -- fine for a debug overlay.
            width_px_resized = width_px / ((scale_r + scale_c) / 2.0)
            self._save_debug_grasp_image(crop_np, candidates, best, env_idx, tag="CROP_CHOSEN",
                                          width_px=width_px_resized)

        return point_w, grasp_quat_w, width_m

    def _save_scene_snapshot(self, cam_data, depth: torch.Tensor, env_idx: int) -> None:
        """Save one undecorated snapshot of the scene from the wrist camera at
        its top-down observe pose -- the camera's raw rgb output as a PNG, plus
        the raw depth array losslessly as .npy. Called once per trial (see the
        regrasp_attempts==0 check at the call site), independent of any debug
        visualization gating.
        """
        out_dir = Path(self.cfg.scene_image_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"trial{self._current_trial}_env{env_idx}_scene"

        depth_np = depth.detach().cpu().numpy()
        np.save(out_dir / f"{stem}_depth.npy", depth_np)

        rgb_out = cam_data.output.get("rgb") if hasattr(cam_data.output, "get") else (
            cam_data.output["rgb"] if "rgb" in cam_data.output else None
        )
        if rgb_out is None:
            print(f"[SCENE] env {env_idx}: no rgb output available (add 'rgb' to grasp_camera.data_types) "
                  f"-- saved depth only to {out_dir / (stem + '_depth.npy')}")
            return
        rgb_np = rgb_out[env_idx, ..., :3].detach().cpu().numpy().astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(rgb_np, mode="RGB").save(out_dir / f"{stem}.png")
            print(f"[SCENE] env {env_idx}: saved raw scene snapshot to {out_dir / (stem + '.png')} "
                  f"(+ depth .npy)")
        except ImportError:
            np.save(out_dir / f"{stem}_rgb.npy", rgb_np)
            print(f"[SCENE] Pillow not installed -- saved raw rgb array to "
                  f"{out_dir / (stem + '_rgb.npy')} instead of a PNG.")

    def _save_sam_debug_image(self, rgb_np, detections: list, matched_box, target_px, env_idx: int) -> None:
        """Save the RGB frame with every SAM proposal (cyan, unlabeled -- SAM has
        no class names, only "something is here", and is no longer used to pick
        the target -- these are shown purely for visual context) and the
        target's PRIVILEGED bounding box (red -- projected directly from its own
        USD AABB + live pose, see _project_object_privileged_box, not a SAM
        guess), plus the target's own projected root position (yellow cross) and
        the true image center (white cross) -- if those two crosses aren't on
        top of each other, the camera geometry is wrong; see the [GEOM] console
        line for the numeric version of this same check.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return
        out_dir = Path(self.cfg.ggcnn_debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(rgb_np, mode="RGB")
        draw = ImageDraw.Draw(img)
        for (x0, y0, x1, y1) in detections:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 255), width=1)
        if matched_box is not None:
            draw.rectangle(list(matched_box), outline=(255, 0, 0), width=3)
        if target_px is not None:
            tx, ty = target_px
            r = 6
            draw.line([tx - r, ty, tx + r, ty], fill=(255, 255, 0), width=2)
            draw.line([tx, ty - r, tx, ty + r], fill=(255, 255, 0), width=2)
            draw.text((tx + 8, ty + 8), f"TARGET: {self._object_infos[0].object_id}", fill=(255, 255, 0))
        w, h = img.size
        draw.line([w / 2 - 10, h / 2, w / 2 + 10, h / 2], fill=(255, 255, 255), width=1)
        draw.line([w / 2, h / 2 - 10, w / 2, h / 2 + 10], fill=(255, 255, 255), width=1)
        path = out_dir / f"trial{self._current_trial}_env{env_idx}_attempt{int(self._regrasp_attempts[env_idx].item())}_SAM.png"
        img.save(path)
        print(f"[SAM] Saved debug image: {path.absolute()} "
              f"(cyan = all SAM proposals (context only), red = target's PRIVILEGED bounding box "
              f"(not a SAM guess), yellow cross = target's projected root position, "
              f"white cross = true image center -- these two crosses should coincide if the camera "
              f"geometry is correct)")

    def _save_debug_grasp_image(
        self, depth_np, candidates: list[dict], chosen: dict | None, env_idx: int, tag: str,
        scale: tuple = (1.0, 1.0), width_px: float | None = None,
    ) -> None:
        """Save the raw depth capture (properly contrast-stretched, not just cast
        to 0-255 -- distance_to_image_plane values are small meter-scale floats
        like 0.3-0.6, which look solid black if you cast them to pixel intensity
        directly without rescaling first) with every candidate grasp point marked,
        and the CHOSEN one highlighted with a line showing its predicted angle.

        If `width_px` is given (the same pixel width value actually sent to the
        gripper -- see the width_px computed right before each call site), also
        draws the grasp rectangle: two short "jaw" lines, `width_px` apart,
        perpendicular to the angle -- i.e. exactly the gap the gripper will try
        to close around at this pose, in the SAME pixel units as the image, so
        you can visually check "does this rectangle actually straddle the
        object" rather than only seeing a center point and a direction line.

        This is the actual visual evidence for whatever the console stats claim --
        look at the saved PNG before trusting (or distrusting) anything else.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            out_dir = Path(self.cfg.ggcnn_debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            attempt = int(self._regrasp_attempts[env_idx].item())
            npy_path = out_dir / f"trial{self._current_trial}_env{env_idx}_attempt{attempt}_{tag}.npy"
            np.save(npy_path, depth_np)
            print(f"[GGCNN] Pillow not installed -- saved raw depth array to {npy_path} instead of a PNG "
                  f"(pip install pillow to get an actual annotated image).")
            return

        out_dir = Path(self.cfg.ggcnn_debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # FIX: previously stretched using THIS capture's own min/max depth --
        # if the frame also caught any far background/off-table area (seen in
        # practice: one capture had min=0.53 max=1.39m), the real ~0.5-0.6m
        # working-height band got compressed into a tiny sliver near the dark
        # end, making the whole image look almost solid black even though
        # nothing was actually wrong with the capture. A tighter-range capture
        # would stretch to a nice full-contrast grey image instead -- same
        # kind of scene, wildly different-looking debug PNGs, purely from
        # normalization. Clip to a fixed working-height band first so every
        # debug image gets consistent, comparable contrast.
        d_lo, d_hi = self.cfg.ggcnn_debug_depth_display_range
        depth_clipped = np.clip(depth_np, d_lo, d_hi)
        if d_hi - d_lo < 1e-6:
            gray = np.zeros_like(depth_np, dtype=np.uint8)
        else:
            gray = ((depth_clipped - d_lo) / (d_hi - d_lo) * 255.0).astype(np.uint8)

        img = Image.fromarray(gray, mode="L").convert("RGB")
        draw = ImageDraw.Draw(img)
        scale_r, scale_c = scale

        for c in candidates:
            row, col = c["row"] * scale_r, c["col"] * scale_c
            r = 3
            draw.ellipse([col - r, row - r, col + r, row + r], outline=(0, 200, 255), width=1)

        drew_rect = False
        if chosen is not None:
            row, col = chosen["row"] * scale_r, chosen["col"] * scale_c
            r = 8
            draw.ellipse([col - r, row - r, col + r, row + r], outline=(255, 0, 0), width=3)
            length = 25
            angle = chosen["angle_rad"]
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            draw.line([col - dx, row - dy, col + dx, row + dy], fill=(0, 255, 0), width=2)

            if width_px is not None:
                # Grasp rectangle: long axis (fixed cosmetic length, same "length"
                # as the green line above) runs along the angle direction; short
                # axis (the actual jaw gap) runs perpendicular to it, spanning
                # width_px -- the real value the gripper closes to at this pose.
                xo, yo = np.cos(angle), np.sin(angle)      # unit vector along angle
                px_, py_ = -np.sin(angle), np.cos(angle)   # unit vector perpendicular
                half_len = length
                half_w = width_px / 2.0
                corners = [
                    (col + half_len * xo - half_w * px_, row + half_len * yo - half_w * py_),
                    (col - half_len * xo - half_w * px_, row - half_len * yo - half_w * py_),
                    (col - half_len * xo + half_w * px_, row - half_len * yo + half_w * py_),
                    (col + half_len * xo + half_w * px_, row + half_len * yo + half_w * py_),
                ]
                draw.polygon(corners, outline=(255, 255, 0), width=2)
                # Highlight the two short "jaw" sides (where the fingers actually
                # contact) more thickly, in magenta, so they stand out from the
                # long cosmetic sides of the rectangle.
                draw.line([corners[0], corners[3]], fill=(255, 0, 255), width=3)
                draw.line([corners[1], corners[2]], fill=(255, 0, 255), width=3)
                drew_rect = True

        attempt = int(self._regrasp_attempts[env_idx].item())
        path = out_dir / f"trial{self._current_trial}_env{env_idx}_attempt{attempt}_{tag}.png"
        img.save(path)
        rect_note = ", yellow/magenta rectangle = gripper jaw width" if drew_rect else ""
        print(f"[GGCNN] Saved debug image: {path.absolute()} "
              f"(red circle + green line = chosen grasp point/angle, cyan = other candidates{rect_note})")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Advance the state machine (not driven by `actions` -- there is no policy)."""
        del actions  # unused: controller is fully scripted from privileged sim state

        target_obj = self.objects[0]
        object_pos_w = target_obj.data.root_pos_w
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        env_origins = self.scene.env_origins

        # ── Safety abort: object dropped off the table, or drifted out of the
        # reachable zone (e.g. shoved there by a clutter collision). Checked every
        # step, regardless of phase, and immediately ends the trial (DONE) so the
        # next episode starts right away rather than the arm chasing a lost object.
        table_top_z = env_origins[:, 2] + self.cfg.table_height
        fell_off = object_pos_w[:, 2] < (table_top_z - self.cfg.object_fall_margin)
        dist_from_center = torch.norm(object_pos_w[:, :2] - env_origins[:, :2], dim=-1)
        out_of_reach = dist_from_center > self.cfg.max_spawn_distance_from_origin
        still_running = (self._phase != PHASE_FREEZE) & (self._phase != PHASE_DONE)
        abort = (fell_off | out_of_reach) & still_running
        if abort.any():
            if self.cfg.enable_debug_state_machine and abort[0]:
                reason = "fell off table" if fell_off[0] else "out of reach"
                print(f"[ABORT] Trial {self._current_trial}: target {reason} -- failing trial")
            self._trial_success = torch.where(abort, torch.zeros_like(self._trial_success), self._trial_success)
            self._phase = torch.where(abort, torch.full_like(self._phase, PHASE_DONE), self._phase)
            self._phase_timer = torch.where(abort, torch.zeros_like(self._phase_timer), self._phase_timer)
            self._gripper_cmd = torch.where(
                abort, torch.full_like(self._gripper_cmd, self.cfg.gripper_open_pos), self._gripper_cmd
            )

        # World-frame TCP target for the *current* phase. Position (and, for
        # DESCEND/CLOSE, orientation) come from the frozen GG-CNN grasp pose
        # computed once at trial start (_capture_and_compute_grasp), not from the
        # object's live position -- same "don't recompute a moving target every
        # step" reasoning as the LIFT-target fix below.
        hover_pos_w = self._grasp_pos_w + torch.tensor(
            [0.0, 0.0, self.cfg.pregrasp_hover_height], device=self.device
        )
        grasp_pos_w = self._grasp_pos_w + torch.tensor(
            [0.0, 0.0, self.cfg.grasp_height_offset], device=self.device
        )
        # LIFT/FREEZE target is anchored to the FROZEN grasp position (captured once,
        # the instant CLOSE finishes -- see close_done handling below), NOT to the
        # object's live position. Once the object is actually being carried upward,
        # its live position rises every step right along with the arm; computing the
        # target from that every step turns "lift_height above where it was grasped" into
        # a receding goalpost the arm can never close the gap on.
        lift_pos_w = self._lift_grasp_pos_w + torch.tensor([0.0, 0.0, self.cfg.lift_height], device=self.device)
        # Fixed vantage point for PHASE_OBSERVE (retry only) -- directly above
        # table center at cfg.observe_height, independent of any candidate grasp,
        # since the whole point is to get the wrist camera a wide-enough view to
        # find one. Same point _move_ee_to_start_pose calibrates at construction.
        observe_pos_w = env_origins.clone()
        observe_pos_w[:, 2] += self.cfg.table_height + self.cfg.observe_height

        phase_tcp_target_w = torch.zeros_like(object_pos_w)
        phase_tcp_target_w[self._phase == PHASE_OBSERVE] = observe_pos_w[self._phase == PHASE_OBSERVE]
        phase_tcp_target_w[self._phase == PHASE_APPROACH] = hover_pos_w[self._phase == PHASE_APPROACH]
        phase_tcp_target_w[self._phase == PHASE_DESCEND] = grasp_pos_w[self._phase == PHASE_DESCEND]
        phase_tcp_target_w[self._phase == PHASE_CLOSE] = grasp_pos_w[self._phase == PHASE_CLOSE]
        phase_tcp_target_w[self._phase == PHASE_LIFT] = lift_pos_w[self._phase == PHASE_LIFT]
        phase_tcp_target_w[self._phase == PHASE_FREEZE] = lift_pos_w[self._phase == PHASE_FREEZE]
        # DONE doesn't use this -- that env is about to be reset (which teleports the
        # robot straight back to its home joint pose, instantly, no scripted motion).

        # Orientation: OBSERVE/APPROACH/DESCEND/CLOSE use the GG-CNN-predicted
        # grasp yaw (self._grasp_quat_w) -- for OBSERVE this is whatever the
        # PREVIOUS (failed) grasp's orientation was, which doesn't matter since
        # OBSERVE only cares about position; LIFT/FREEZE keep whatever
        # orientation the object was actually grasped at (frozen alongside
        # _lift_grasp_pos_w) rather than snapping back to a neutral orientation
        # mid-lift.
        phase_target_quat_w = self._grasp_quat_w.clone()
        phase_target_quat_w[self._phase == PHASE_LIFT] = self._lift_grasp_quat_w[self._phase == PHASE_LIFT]
        phase_target_quat_w[self._phase == PHASE_FREEZE] = self._lift_grasp_quat_w[self._phase == PHASE_FREEZE]

        hand_target_pos_w = self._tcp_target_to_hand_target(phase_tcp_target_w, phase_target_quat_w)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, hand_target_pos_w, phase_target_quat_w
        )
        self._target_pos_b = target_pos_b
        self._target_quat_b = target_quat_b
        self.ik_controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))

        # Gripper command per phase: open during observe/approach/descend, closed to
        # the GG-CNN-predicted width (or fully closed, for the naive fallback, since
        # its fallback width is set to the fully-open value and clamped down at
        # CLOSE) while carrying the object. (Retries below can still override this
        # back open for envs that just failed a grasp check.)
        self._gripper_cmd = torch.where(
            (self._phase == PHASE_OBSERVE) | (self._phase == PHASE_APPROACH) | (self._phase == PHASE_DESCEND),
            torch.full_like(self._gripper_cmd, self.cfg.gripper_open_pos),
            self._gripper_cmd,
        )
        closing = self._phase == PHASE_CLOSE
        if closing.any():
            close_target = torch.clamp(self._grasp_width_m / 2.0, self.cfg.gripper_closed_pos, self.cfg.gripper_open_pos)
            frac = torch.clamp(self._phase_timer.float() / self.cfg.grasp_close_steps, 0.0, 1.0)
            interp = self.cfg.gripper_open_pos + frac * (close_target - self.cfg.gripper_open_pos)
            self._gripper_cmd = torch.where(closing, interp, self._gripper_cmd)
        holding_closed = (self._phase == PHASE_LIFT) | (self._phase == PHASE_FREEZE)
        hold_width = torch.clamp(self._grasp_width_m / 2.0, self.cfg.gripper_closed_pos, self.cfg.gripper_open_pos)
        self._gripper_cmd = torch.where(holding_closed, hold_width, self._gripper_cmd)

        # Phase transitions.
        pos_err = torch.norm(ee_pos_b - target_pos_b, dim=-1)
        reached = pos_err < self.cfg.pose_reach_pos_tol
        self._phase_timer += 1

        observe_done = (self._phase == PHASE_OBSERVE) & (reached | (self._phase_timer > self.cfg.observe_timeout_steps))
        approach_done = (self._phase == PHASE_APPROACH) & (reached | (self._phase_timer > self.cfg.approach_timeout_steps))
        descend_done = (self._phase == PHASE_DESCEND) & (reached | (self._phase_timer > self.cfg.descend_timeout_steps))
        close_done = (self._phase == PHASE_CLOSE) & (self._phase_timer > self.cfg.grasp_close_steps)
        if close_done.any():
            # Snapshot the grasp point NOW, before any lifting happens -- this is
            # what the LIFT/FREEZE target above is anchored to, instead of the
            # object's live (and, once lifted, continuously rising) position.
            self._lift_grasp_pos_w = torch.where(
                close_done.unsqueeze(-1), grasp_pos_w, self._lift_grasp_pos_w
            )
            self._lift_grasp_quat_w = torch.where(
                close_done.unsqueeze(-1), phase_target_quat_w, self._lift_grasp_quat_w
            )
        # "Reached" the lift target -- checked (and acted on) immediately, no waiting
        # period, so a successful trial ends the moment the height is hit.
        lift_reached = (self._phase == PHASE_LIFT) & (reached | (self._phase_timer > self.cfg.lift_timeout_steps))
        # Reach height -> freeze -> reset, with nothing in between: once the freeze
        # hold is done, the episode ends immediately (DONE => terminated next step),
        # and _reset_idx teleports the robot back to home as part of that reset.
        freeze_done = (self._phase == PHASE_FREEZE) & (self._phase_timer > self.cfg.freeze_steps)

        # On a misgrasp retry, the arm was sent back up through PHASE_OBSERVE with
        # _pending_recapture set (see can_retry below) instead of straight back to
        # DESCEND. THIS is where "move the arm up, capture, and relocate the
        # gripper" actually happens: once it's reached the vantage point,
        # recapture + recompute a (possibly different) candidate grasp before
        # falling through to the normal observe_done -> APPROACH transition below,
        # which will use the freshly updated self._grasp_pos_w starting next step.
        recapture_now = observe_done & self._pending_recapture
        if recapture_now.any():
            recapture_ids = recapture_now.nonzero(as_tuple=True)[0]
            paused = self._capture_and_compute_grasp(recapture_ids)
            self._trial_paused_s[recapture_ids] += paused
            self._pending_recapture = torch.where(
                recapture_now, torch.zeros_like(self._pending_recapture), self._pending_recapture
            )
            # "Never fall back to naive" -- if this recapture STILL had to resort
            # to naive (degenerate depth, or nothing usable even after relaxing
            # the quality gate within the target mask), don't accept it: treat
            # this as another failed attempt and loop straight back into another
            # recapture, consuming budget from the same max_regrasp_attempts pool
            # used for physical grasp failures. Only once that budget is
            # genuinely exhausted does the naive pose actually get used.
            still_naive = recapture_now & self._used_naive_fallback
            can_retry_vision = still_naive & (self._regrasp_attempts < self.cfg.max_regrasp_attempts)
            if can_retry_vision.any():
                self._regrasp_attempts = torch.where(
                    can_retry_vision, self._regrasp_attempts + 1, self._regrasp_attempts
                )
                self._pending_recapture = torch.where(
                    can_retry_vision, torch.ones_like(self._pending_recapture), self._pending_recapture
                )
                # Exclude these envs from the observe_done -> APPROACH transition
                # below -- they stay in PHASE_OBSERVE for another recapture loop
                # next step (the arm hasn't moved, so `reached` will still be
                # True and observe_done will fire again almost immediately).
                observe_done = observe_done & (~can_retry_vision)
                if self.cfg.enable_debug_state_machine and can_retry_vision[0]:
                    print(f"[RETRY-VISION] Trial {self._current_trial}: naive fallback rejected -- "
                          f"retrying capture (attempt {int(self._regrasp_attempts[0].item())}/"
                          f"{self.cfg.max_regrasp_attempts})")
            exhausted_vision = still_naive & (~can_retry_vision)
            if exhausted_vision.any():
                print(f"[GGCNN] WARNING: retry budget exhausted while still only able to produce a naive "
                      f"grasp for {int(exhausted_vision.sum())} env(s) -- proceeding with naive as a "
                      f"genuine last resort (no vision-derived point was ever found on the target).")

        for done_mask, next_phase in [
            (observe_done, PHASE_APPROACH),
            (approach_done, PHASE_DESCEND), (descend_done, PHASE_CLOSE), (close_done, PHASE_LIFT),
            (freeze_done, PHASE_DONE),
        ]:
            if done_mask.any():
                self._phase = torch.where(done_mask, torch.full_like(self._phase, next_phase), self._phase)
                self._phase_timer = torch.where(done_mask, torch.zeros_like(self._phase_timer), self._phase_timer)

        # ── Grasp verification, the instant the lift target is reached ──────────
        # Compare the object's own height gain to the end effector's: if the arm is
        # up but the *target* didn't come with it (empty gripper, or a clutter
        # object got grasped instead while the target stayed on the table), that's a
        # failed grasp -- release and redescend for another attempt rather than
        # ending the trial. Only give up (and fail the trial) after
        # max_regrasp_attempts.
        if lift_reached.any():
            object_height_gain = object_pos_w[:, 2] - self._object_init_pos[:, 2]
            target_lifted = object_height_gain >= (self.cfg.lift_height * self.cfg.success_lift_fraction)

            succeeded = lift_reached & target_lifted
            can_retry = lift_reached & (~target_lifted) & (self._regrasp_attempts < self.cfg.max_regrasp_attempts)
            exhausted = lift_reached & (~target_lifted) & (self._regrasp_attempts >= self.cfg.max_regrasp_attempts)

            self._trial_success = torch.where(succeeded, torch.ones_like(self._trial_success), self._trial_success)

            # Success freezes in place (held pose, gripper still closed) for
            # freeze_steps, then ends the trial immediately -- no scripted return
            # motion. A used-up trial (exhausted retries) ends immediately too,
            # skipping the freeze -- nothing to show off there.
            self._phase = torch.where(succeeded, torch.full_like(self._phase, PHASE_FREEZE), self._phase)
            self._phase_timer = torch.where(succeeded, torch.zeros_like(self._phase_timer), self._phase_timer)
            self._phase = torch.where(exhausted, torch.full_like(self._phase, PHASE_DONE), self._phase)
            self._phase_timer = torch.where(exhausted, torch.zeros_like(self._phase_timer), self._phase_timer)

            if can_retry.any():
                self._regrasp_attempts = torch.where(can_retry, self._regrasp_attempts + 1, self._regrasp_attempts)
                # Move the arm back UP to the observation vantage point (not
                # straight to DESCEND, and not just APPROACH's low hover -- the
                # wrist camera needs the same wide view it started with to find a
                # new candidate) and flag it to recapture once it gets there --
                # see the recapture_now handling above, which is where that
                # actually runs.
                self._phase = torch.where(can_retry, torch.full_like(self._phase, PHASE_OBSERVE), self._phase)
                self._phase_timer = torch.where(can_retry, torch.zeros_like(self._phase_timer), self._phase_timer)
                self._pending_recapture = torch.where(
                    can_retry, torch.ones_like(self._pending_recapture), self._pending_recapture
                )
                self._gripper_cmd = torch.where(
                    can_retry, torch.full_like(self._gripper_cmd, self.cfg.gripper_open_pos), self._gripper_cmd
                )
                if self.cfg.enable_debug_state_machine and can_retry[0]:
                    print(f"[RETRY] Trial {self._current_trial}: target not lifted with the gripper "
                          f"(attempt {int(self._regrasp_attempts[0].item())}/{self.cfg.max_regrasp_attempts}) "
                          f"-- moving up to recapture and relocate")

        if self.cfg.enable_debug_state_machine:
            p = int(self._phase[0].item())
            print(f"[SM] phase={_PHASE_NAMES[p]:9s} pos_err={pos_err[0].item():.4f} "
                  f"gripper={self._gripper_cmd[0].item():.3f}")

    def _apply_action(self) -> None:
        """Run one IK solve + write joint targets. Called every decimated substep."""
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._arm_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()

        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        all_joint_pos_target = self.robot.data.joint_pos_target.clone()
        all_joint_pos_target[:, self._arm_joint_ids] = joint_pos_des
        all_joint_pos_target[:, self._gripper_joint_ids] = self._gripper_cmd.unsqueeze(-1)

        self.robot.set_joint_position_target(all_joint_pos_target)
        self.robot.write_data_to_sim()

    # ─────────────────────────────────────────────────────────────────────
    # Gym plumbing
    # ─────────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # No policy consumes this -- kept only because DirectRLEnv expects it.
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._phase == PHASE_DONE
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Record the outcome of the trial that just ended (skip the very first reset,
        # where nothing has run yet).
        if self._current_trial > 0 or self._phase[env_ids].eq(PHASE_DONE).any():
            for i in env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids):
                start_time = self._trial_start_time[i]
                # picking_time counts only ACTIVE movement -- time spent capturing
                # the top-down image and running GG-CNN (both the initial capture
                # and any retry recaptures) is excluded via _trial_paused_s, which
                # _capture_and_compute_grasp's callers accumulate into.
                if start_time is not None:
                    picking_time = (time.time() - start_time) - float(self._trial_paused_s[i].item())
                    picking_time = max(picking_time, 0.0)
                else:
                    picking_time = 0.0
                self._trial_results.append({
                    "trial": self._current_trial,
                    "object_id": self._object_infos[0].object_id,
                    "success": bool(self._trial_success[i].item()),
                    # Matches BenchmarkEnv's trial_result fields: "drops" is grasp
                    # attempts that reached lift height without the target coming
                    # up, each one released and redescended for another try (see
                    # can_retry in _pre_physics_step); "picking_time" is wall-clock
                    # seconds of active movement only (image capture/inference time
                    # excluded), same spirit as BenchmarkEnv's time.time()-based
                    # measurement.
                    "drops": int(self._regrasp_attempts[i].item()),
                    "picking_time": picking_time,
                })
        self._current_trial += 1

        # Reset robot to its start pose (EE cfg.observe_height above table
        # center) in one instant joint-state write. No convergence motion here --
        # that IK solve only ever runs once, at construction time (see
        # _start_arm_joint_pos in __init__), so every trial starts there
        # immediately instead of visibly traveling to it each time.
        target_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        target_joint_pos[:, self._arm_joint_ids] = self._start_arm_joint_pos[env_ids]
        target_joint_pos[:, self._gripper_joint_ids] = self.cfg.gripper_open_pos
        target_joint_vel = torch.zeros_like(target_joint_pos)
        self.robot.write_joint_state_to_sim(target_joint_pos, target_joint_vel, env_ids=env_ids)
        # write_joint_state_to_sim only sets the physical joint state -- it does NOT
        # touch the PD position-drive TARGET, which is a separate, persistent PhysX
        # command. If the arm was mid-motion or stuck in an awkward pose right
        # before this reset, the drive target left over from that last
        # _apply_action call would otherwise still be pointing at that old
        # position, and the very next physics step would immediately start
        # pulling the arm back toward it -- looking exactly like "reset didn't
        # fully take." Explicitly re-pointing the drive target here at the same
        # reset pose fixes that.
        self.robot.set_joint_position_target(target_joint_pos, env_ids=env_ids)
        self.robot.write_data_to_sim()
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)

        self.ik_controller.reset(env_ids)
        self._phase[env_ids] = PHASE_APPROACH
        self._phase_timer[env_ids] = 0
        self._gripper_cmd[env_ids] = self.cfg.gripper_open_pos
        self._trial_success[env_ids] = False
        self._regrasp_attempts[env_ids] = 0
        self._lift_grasp_pos_w[env_ids] = 0.0
        self._lift_grasp_quat_w[env_ids] = self._down_quat_w[env_ids]
        self._trial_paused_s[env_ids] = 0.0
        self._pending_recapture[env_ids] = False
        for i in env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids):
            self._trial_start_time[i] = time.time()

        # Re-select and re-spawn the target (+ clutter) for the next trial.
        self._select_objects_to_spawn()
        self._initialize_scene_objects(env_ids)

        # Capture once, before the robot starts moving -- the arm is already
        # sitting at its start pose (just written above), not yet driven anywhere
        # by the state machine, since that only happens starting next env.step().
        paused = self._capture_and_compute_grasp(env_ids)
        self._trial_paused_s[env_ids] += paused

        # "Never fall back to naive" -- if even the very first capture had to
        # resort to naive (degenerate depth, or nothing usable on the target
        # even after relaxing the quality gate), don't start the trial with
        # it: route straight into the same PHASE_OBSERVE -> recapture retry
        # loop used for a failed physical attempt (see _pre_physics_step's
        # recapture_now handling), consuming budget from max_regrasp_attempts,
        # instead of ever proceeding into APPROACH with a naive pose.
        env_ids_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), device=self.device)
        naive_mask = self._used_naive_fallback[env_ids_t]
        if naive_mask.any():
            naive_ids = env_ids_t[naive_mask]
            can_retry_mask = self._regrasp_attempts[naive_ids] < self.cfg.max_regrasp_attempts
            retry_ids = naive_ids[can_retry_mask]
            if retry_ids.numel() > 0:
                self._regrasp_attempts[retry_ids] += 1
                self._phase[retry_ids] = PHASE_OBSERVE
                self._pending_recapture[retry_ids] = True
                print(f"[GGCNN] Trial {self._current_trial}: initial capture used naive fallback for "
                      f"{int(retry_ids.numel())} env(s) -- routing into a retry rather than starting the "
                      f"trial with it (attempt {int(self._regrasp_attempts[retry_ids[0]].item())}/"
                      f"{self.cfg.max_regrasp_attempts})")
            exhausted_ids = naive_ids[~can_retry_mask]
            if exhausted_ids.numel() > 0:
                print(f"[GGCNN] WARNING: retry budget exhausted for {int(exhausted_ids.numel())} env(s) "
                      f"at trial start -- proceeding with naive as a genuine last resort.")

    def get_last_trial_result(self) -> dict | None:
        return self._trial_results[-1] if self._trial_results else None

    def get_all_trial_results(self) -> list[dict]:
        return list(self._trial_results)


def _euler_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """roll/pitch/yaw (N,3) -> quaternion (N,4) in (w, x, y, z) order."""
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)