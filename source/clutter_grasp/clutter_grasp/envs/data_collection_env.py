# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
DataCollectionEnv
=================
Lightweight Isaac Lab env: table + random objects + camera. No robot.

Key design for variable object count per scene
----------------------------------------------
  USD prims are fixed at scene creation time, so we always spawn
  max_objects_to_spawn prims. Between scenes we:
    - Randomly pick N objects from available_objects (N varies each scene)
    - Teleport unused prims far below the table (hidden, no physics effect)
    - Randomise active object positions + orientations on the table
    - (camera jitter hook is a no-op; image-level augmentation is done by the
      collector script classifier/isaac/collect_classifier_dataset.py)

Objects are read from ``cfg.object_usd_dir`` (default ``data/egad_usd``).
Registered as gym ID ``ClutterGrasp-DataCollection-v0``.

This gives a different number of visible objects and different object
types every scene, without recreating the env.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

import omni.usd
from pxr import UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .data_collection_env_cfg import DataCollectionEnvCfg, ObjectSpawnInfo


# ── Math utilities ────────────────────────────────────────────────────────────

@torch.jit.script
def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    cy, sy = torch.cos(yaw * 0.5),   torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5),  torch.sin(roll * 0.5)
    return torch.stack([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ], dim=-1)


def quat_apply_batch(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """(B,4) × (B,N,3) → (B,N,3)."""
    q   = quat.unsqueeze(1).expand(-1, vec.shape[1], -1)
    w   = q[..., 0:1]
    xyz = q[..., 1:4]
    t   = 2 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────

class DataCollectionEnv(DirectRLEnv):
    """Table + randomly placed objects + camera. No robot."""

    cfg: DataCollectionEnvCfg

    def __init__(self, cfg: DataCollectionEnvCfg,
                 render_mode: str | None = None, **kwargs):
        self.cfg = cfg

        # Always allocate max_objects slots — subset activated per scene
        self._max_slots           = cfg.max_objects_to_spawn
        self.objects: list[RigidObject]           = []
        self._all_object_infos: list[ObjectSpawnInfo] = []  # pool of available objects
        self._object_infos: list[ObjectSpawnInfo]     = []  # active this scene
        self._active_indices: list[int]               = []  # which slots are active
        self._object_mesh_points_local: torch.Tensor | None = None

        # Picking order state
        self._current_object_idx = 0
        self._picking_order: list[int] = []

        self.camera: TiledCamera | None = None

        # Load available objects
        self._load_available_objects()

        # Call super (sets up scene, starts sim)
        super().__init__(cfg, render_mode, **kwargs)

        # Retrieve camera
        if "data_camera" in self.scene.sensors:
            self.camera = self.scene.sensors["data_camera"]
            print("[DataCollectionEnv] ✓ Camera retrieved")
        else:
            print("[DataCollectionEnv] ✗ Camera not found — check --enable_cameras")

        # First scene setup
        self.respawn_scene()
        print(f"[DataCollectionEnv] Ready")

    # ═════════════════════════════════════════════════════════════════════
    # DirectRLEnv stubs
    # ═════════════════════════════════════════════════════════════════════

    def _setup_scene(self):
        """
        Create max_objects_to_spawn USD prims.
        All prims are always present — inactive ones are hidden below table.
        """
        print(f"\n[DataCollectionEnv] Creating {self._max_slots} object slots...")

        # Use a random selection for initial USD types (will be overridden in respawn)
        initial_selection = np.random.choice(
            len(self._all_object_infos), self._max_slots, replace=True
        )

        for i in range(self._max_slots):
            obj_info = self._all_object_infos[initial_selection[i]]
            obj_cfg  = RigidObjectCfg(
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
                    pos=(0.0, 0.0, self.cfg.table_height + 0.05),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            obj = RigidObject(obj_cfg)
            self.objects.append(obj)
            self.scene.rigid_objects[f"object_{i}"] = obj

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self._create_table_in_source()
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Lighting with slight randomisable intensity
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Camera
        cam = TiledCamera(self.cfg.camera)
        self.scene.sensors["data_camera"] = cam
        print(f"[DataCollectionEnv] _setup_scene complete ({self._max_slots} slots)")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return d, d

    def _reset_idx(self, env_ids):
        pass

    # ═════════════════════════════════════════════════════════════════════
    # Table geometry
    # ═════════════════════════════════════════════════════════════════════

    def _create_table_in_source(self):
        src   = "/World/envs/env_0"
        leg_h = self.cfg.table_height - self.cfg.table_thickness

        top_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.table_width, self.cfg.table_depth, self.cfg.table_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        top_cfg.func(
            f"{src}/Table/TableTop", top_cfg,
            translation=(0.0, 0.0, self.cfg.table_height - self.cfg.table_thickness / 2)
        )
        ox = self.cfg.table_width  / 2 - self.cfg.leg_radius - 0.02
        oy = self.cfg.table_depth  / 2 - self.cfg.leg_radius - 0.02
        for i, (lx, ly) in enumerate([(ox,oy),(-ox,oy),(ox,-oy),(-ox,-oy)]):
            leg_cfg = sim_utils.CylinderCfg(
                radius=self.cfg.leg_radius, height=leg_h,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=False, kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            leg_cfg.func(f"{src}/Table/Leg{i}", leg_cfg, translation=(lx, ly, leg_h/2))

    # ═════════════════════════════════════════════════════════════════════
    # Object pool
    # ═════════════════════════════════════════════════════════════════════

    def _load_available_objects(self):
        usd_dir = Path(self.cfg.object_usd_dir)
        if not usd_dir.exists():
            raise FileNotFoundError(
                f"USD dir not found: {usd_dir}\n"
                "Download the EGAD meshes from https://dougsm.github.io/egad/ and convert them to USD "
                "(see classifier/README.md), or set object_usd_dir.")
        for f in sorted(usd_dir.glob("*.usd")):
            self._all_object_infos.append(
                ObjectSpawnInfo(object_id=f.stem, usd_path=str(f))
            )
        if not self._all_object_infos:
            raise RuntimeError(f"No USD files in {usd_dir}")
        print(f"[DataCollectionEnv] Found {len(self._all_object_infos)} objects in {usd_dir}")

    # ═════════════════════════════════════════════════════════════════════
    # Scene respawn — new object count, new objects, new positions every call
    # ═════════════════════════════════════════════════════════════════════

    def respawn_scene(self):
        """
        Randomise everything for a new scene:
          - Random N objects (min_objects_to_spawn to max_objects_to_spawn)
          - Random object types from the full pool
          - Random positions + orientations on table
          - Slight camera pose jitter (domain randomisation)
          - Inactive slots teleported below table (invisible)
        """

        # ── 1. Pick random N and random object types ──────────────────────
        n_active = np.random.randint(
            self.cfg.min_objects_to_spawn,
            self.cfg.max_objects_to_spawn + 1,
        )
        chosen_idx   = np.random.choice(
            len(self._all_object_infos), size=n_active, replace=True
        )
        self._object_infos   = [self._all_object_infos[i] for i in chosen_idx]
        self._active_indices = list(range(n_active))
        self._picking_order  = list(range(n_active))

        print(f"\n[SPAWN] {n_active} objects this scene: "
              f"{[o.object_id for o in self._object_infos]}")

        env_origins = self.scene.env_origins
        margin = self.cfg.spawn_area_margin
        uw = self.cfg.table_width  - 2 * margin
        ud = self.cfg.table_depth  - 2 * margin
        positions = self._generate_random_positions(n_active, uw, ud)

        for i in range(self._max_slots):
            obj = self.objects[i]
            obj.update(dt=self.cfg.sim.dt)

            if i < n_active:
                # ── Active: place on table ────────────────────────────────
                pos_w         = env_origins.clone()
                pos_w[:, 0]  += positions[i][0]
                pos_w[:, 1]  += positions[i][1]
                pos_w[:, 2]  += self.cfg.table_height + 0.05

                roll  = (torch.rand(self.num_envs, device=self.device)*2 - 1) * np.pi
                pitch = (torch.rand(self.num_envs, device=self.device)*2 - 1) * np.pi
                yaw   = (torch.rand(self.num_envs, device=self.device)*2 - 1) * np.pi
                euler = torch.stack([roll, pitch, yaw], dim=-1)
                quat  = euler_to_quaternion(euler)
                quat  = quat / torch.norm(quat, dim=-1, keepdim=True)
            else:
                # ── Inactive: hide 5m below table ────────────────────────
                pos_w        = env_origins.clone()
                pos_w[:, 2] -= 5.0
                quat = torch.tensor(
                    [[1., 0., 0., 0.]], device=self.device
                ).expand(self.num_envs, -1)

            state         = obj.data.default_root_state.clone()
            state[:, 0:3] = pos_w
            state[:, 3:7] = quat
            state[:, 7:]  = 0.0
            obj.write_root_state_to_sim(state)

        # ── 2. Camera domain randomisation ────────────────────────────────
        self._jitter_camera()

        # ── 3. Settle ─────────────────────────────────────────────────────
        print(f"[SPAWN] Settling {self.cfg.spawn_settling_steps} steps...")
        for step in range(self.cfg.spawn_settling_steps):
            self.sim.step(render=False)
            if step % 50 == 0:
                for obj in self.objects:
                    obj.update(dt=self.cfg.sim.dt)

        for obj in self.objects:
            obj.update(dt=self.cfg.sim.dt)

        # ── 4. Final render + camera update ───────────────────────────────
        if self.camera is not None:
            for _ in range(5):
                self.sim.step(render=True)
            self.camera.update(dt=self.cfg.sim.dt)

        print(f"[SPAWN] ✓ Scene ready — {n_active} active objects")

    def _jitter_camera(self):
        """
        Camera jitter is not applied at runtime — TiledCamera bakes its
        viewport at sensor creation and ignores USD prim moves afterward.
        Domain randomisation is applied at the image level in the collector
        (brightness, contrast, flip) which is sufficient for generalisation.
        """
        pass

    # ═════════════════════════════════════════════════════════════════════
    # Position generation
    # ═════════════════════════════════════════════════════════════════════

    def _generate_random_positions(self, n: int, uw: float, ud: float) -> list:
        """Random positions with minimum spacing, grid fallback."""
        positions = []
        for i in range(n):
            placed = False
            for _ in range(150):
                x = (np.random.rand() - 0.5) * uw
                y = (np.random.rand() - 0.5) * ud
                if not positions or min(
                    np.sqrt((x-px)**2 + (y-py)**2) for px, py in positions
                ) >= self.cfg.min_object_spacing:
                    positions.append((x, y))
                    placed = True
                    break
            if not placed:
                cols = max(1, int(np.ceil(np.sqrt(n))))
                col  = i % cols
                row  = i // cols
                rows = max(1, int(np.ceil(n / cols)))
                x    = (col / max(cols-1, 1) - 0.5) * uw * 0.85
                y    = (row / max(rows-1, 1) - 0.5) * ud * 0.85
                positions.append((x, y))
        return positions

    # ═════════════════════════════════════════════════════════════════════
    # Point cloud extraction (identical to MultiObjectSequentialEnv)
    # Only called for active objects
    # ═════════════════════════════════════════════════════════════════════

    def _extract_object_mesh_for_current(self, obj_idx: int):
        """Extract mesh for active obj_idx. Returns numpy (P,3) or fallback."""
        obj_info  = self._object_infos[obj_idx]
        stage     = omni.usd.get_context().get_stage()
        env_path  = self.scene.env_prim_paths[0]
        prim_path = f"{env_path}/Object_{obj_idx}"
        obj_prim  = stage.GetPrimAtPath(prim_path)

        if not obj_prim.IsValid():
            print(f"[MESH] Prim not valid: {prim_path}")
            return self._generate_default_point_cloud()

        all_vertices = []

        def collect(prim, parent_xform=Gf.Matrix4d(1.0)):
            xform = parent_xform
            if prim.IsA(UsdGeom.Xformable):
                xform = parent_xform * UsdGeom.Xformable(prim).GetLocalTransformation()
            if prim.IsA(UsdGeom.Mesh):
                pts_attr = UsdGeom.Mesh(prim).GetPointsAttr()
                if pts_attr and pts_attr.Get():
                    verts = np.array(
                        [[float(p[0]), float(p[1]), float(p[2])]
                         for p in pts_attr.Get()]
                    )
                    if xform != Gf.Matrix4d(1.0):
                        xm    = np.array([[xform[r][c] for c in range(4)]
                                          for r in range(4)])
                        verts = (xm @ np.hstack(
                            [verts, np.ones((len(verts), 1))]
                        ).T).T[:, :3]
                    all_vertices.append(verts)
            for child in prim.GetChildren():
                collect(child, xform)

        collect(obj_prim)

        if not all_vertices:
            return self._generate_default_point_cloud()

        verts = np.vstack(all_vertices)
        max_c = np.abs(verts).max()
        if   max_c > 10.0: verts *= 0.001
        elif max_c > 1.0:  verts *= 0.01

        pts = self._subsample_vertices(verts, self.cfg.num_object_pc_points)
        self._object_mesh_points_local = torch.tensor(
            pts, device=self.device, dtype=torch.float32
        )
        print(f"[MESH] ✓ {len(pts)} pts — {obj_info.object_id}")
        return pts

    def _subsample_vertices(self, verts: np.ndarray, n: int) -> np.ndarray:
        if len(verts) > n:
            return verts[np.random.choice(len(verts), n, replace=False)]
        elif len(verts) < n:
            extra = np.tile(verts, (n // len(verts) + 1, 1))[:n]
            return extra + np.random.normal(0, 0.0005, extra.shape)
        return verts

    def _generate_default_point_cloud(self) -> np.ndarray:
        n, r = self.cfg.num_object_pc_points, 0.03
        phi  = np.pi * (3. - np.sqrt(5.))
        pts  = []
        for i in range(n):
            y  = 1 - (i / float(n - 1)) * 2
            ry = np.sqrt(max(1 - y*y, 0))
            th = phi * i
            pts.append([np.cos(th)*ry*r, y*r, np.sin(th)*ry*r])
        pts_arr = np.array(pts)
        self._object_mesh_points_local = torch.tensor(
            pts_arr, device=self.device, dtype=torch.float32
        )
        return pts_arr

    # ═════════════════════════════════════════════════════════════════════
    # Camera capture
    # ═════════════════════════════════════════════════════════════════════

    def capture_rgb(self) -> np.ndarray | None:
        if self.camera is None:
            return None
        try:
            self.camera.update(dt=self.cfg.sim.dt)
            rgb = self.camera.data.output["rgb"][0].cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)
            return rgb
        except Exception as e:
            print(f"[DataCollectionEnv] capture_rgb failed: {e}")
            return None