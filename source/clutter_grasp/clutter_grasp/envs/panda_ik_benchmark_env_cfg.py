# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Panda + parallel-gripper, privileged-IK pick-and-lift benchmark.

This is a sibling of ``benchmark_env_cfg.py`` / ``BenchmarkEnvCfg``, NOT a subclass of it.
The ContactileHand env controls a *free-floating* hand root directly (root velocity
control, no arm). A Panda is a fixed-base 7-DOF arm, so the controller stack is
fundamentally different (Jacobian-based IK vs. direct root pose control) even though
the table / object-spawning / trial-loop logic is identical in spirit.

No learning is involved in the CONTROLLER (the arm is driven by a scripted
approach -> descend -> grasp -> lift state machine via differential IK, not a
trained policy). Grasp POSE SELECTION, however, is learned: a top-down camera +
GG-CNN predicts where and at what angle/width to grasp the target object, instead
of always grasping at the object's raw root position with a fixed orientation.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

try:
    # Ships with isaaclab_assets; high-PD gains track IK targets much more cleanly
    # than the default compliant gains (which are tuned for RL torque-ish control).
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG as _FRANKA_CFG
except ImportError:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG as _FRANKA_CFG


# Same asset layout as your existing benchmark experiments.
from clutter_grasp.paths import OBJECTS_DIR
USD_DIR = str(OBJECTS_DIR)


@dataclass
class ObjectSpawnInfo:
    """Object to spawn. No policy fields needed -- the controller is scripted, not learned."""
    object_id: str
    usd_path: str


# Same clearances as the protocol's CLUTTER_CONFIGS (clutter_grasp.protocol.clutter_levels), but a
# fixed neighbour count per level (4 / 6 / 8) and no classifier check -- this is how the Panda IK
# baseline results in the paper were produced.
CLUTTER_CONFIGS = {
    'C0_easy': {
        'num_neighbors': (4, 4),
        'min_clearance': 0.10,
        'max_clearance': 0.20,
        'description': 'Easy - Few isolated neighbors',
    },
    'C1_medium': {
        'num_neighbors': (6, 6),
        'min_clearance': 0.08,
        'max_clearance': 0.12,
        'description': 'Medium - Moderate clutter',
    },
    'C2_hard': {
        'num_neighbors': (8, 8),
        'min_clearance': 0.03,
        'max_clearance': 0.08,
        'description': 'Hard - Dense clutter',
    },
}


@configclass
class PandaIKBenchmarkEnvCfg(DirectRLEnvCfg):
    """Configuration for the Panda / parallel-gripper / privileged-IK benchmark."""

    # ── Simulation ──────────────────────────────────────────────────────────
    decimation = 2
    # One episode == one full trial (approach/descend/grasp/lift/verify/retry/return).
    # Sized to the WORST CASE for max_regrasp_attempts=19 (20 total attempts),
    # using every phase's own timeout below -- not the original
    # BenchmarkEnvCfg.max_trial_timesteps=1000 figure, which only budgeted for
    # ~3 attempts and was silently truncating (forced reset via DirectRLEnv's
    # own episode_length_buf >= max_episode_length) long before
    # max_regrasp_attempts ever got used up -- raising the retry count alone
    # does nothing without also raising this budget to match.
    # First attempt (no retry-observe travel):  approach(150)+descend(150)+close(40)+lift(300) = 640
    # Each of the 19 retries: observe(150)+approach(150)+descend(150)+close(40)+lift(300)  = 790
    # 640 + 19*790 = 15650, +freeze(20) for a successful final attempt = 15670 -> round up to 16000.
    # 16000 steps * decimation(2) * sim.dt(1/120) = 266.7s. This is a ceiling, not a typical
    # duration -- most attempts finish well under their phase timeouts in practice.
    episode_length_s = 16000 * 2 * (1 / 120)

    # Gym plumbing only -- there is no learned policy, the env is driven internally
    # by the scripted state machine, not by env.step(actions).
    action_space = 1
    observation_space = 1
    state_space = 0
    asymmetric_obs = False

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # single env for sequential trials, same as BenchmarkEnvCfg
        env_spacing=2.5,
        replicate_physics=False,
    )

    # ── Table (identical to BenchmarkEnvCfg) ───────────────────────────────
    table_width: float = 0.85
    table_depth: float = 0.85
    table_height: float = 0.8
    table_thickness: float = 0.05
    leg_radius: float = 0.03

    # ── Robot: fixed-base Panda, floating just outside the table edge ───────
    # Base sits at z = table_height (so it's still working at tabletop level), but
    # OUTSIDE the tabletop's own footprint by robot_edge_gap -- not inset onto the
    # table surface. This keeps the base clear of the table's collision geometry
    # (a very small inset was interpenetrating the tabletop, which fights the
    # IK-driven motion and can make the arm never actually reach anything) and
    # clears the near-base dead zone objects could otherwise spawn into.
    robot_edge_gap: float = 0.05  # meters the base floats past the table edge
    robot_cfg: ArticulationCfg = _FRANKA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    # NOTE: init_state.pos is set at runtime in the env (needs table_width/2 and
    # table_height), since @configclass dataclass fields can't reference sibling
    # fields at class-build time.

    ee_body_name: str = "panda_hand"
    gripper_joint_names: tuple = ("panda_finger_joint1", "panda_finger_joint2")
    gripper_open_pos: float = 0.04    # meters, per finger
    gripper_closed_pos: float = 0.0
    # Offset from panda_hand frame to the actual fingertip contact point (TCP),
    # along the hand's local +z (approach axis). Standard Franka panda_hand->TCP offset.
    tcp_offset_z: float = 0.1034

    # ── Differential IK controller ─────────────────────────────────────────
    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},
    )
    # Max joint-space step per physics step fed to the IK controller's target
    # (position units: m/step for the underlying pose interpolation cap below).
    max_ee_lin_speed: float = 0.35   # m/s cap on commanded end-effector translation
    max_ee_ang_speed: float = 2.0    # rad/s cap on commanded end-effector rotation

    # ── Observation pose (also the wrist camera's vantage point) ──────────────
    # At the start of each trial, the arm goes to a fixed pose directly above the
    # table center, high enough that the wrist-mounted camera's FOV covers a
    # useful chunk of the table (not just a close-up of whatever's directly
    # beneath the gripper) -- 0.10m (right down at the table) was fine for the
    # old fixed table-mounted camera design, but far too low for this one.
    # Reused for BOTH the initial per-trial vantage point (instant teleport, at
    # reset -- see _start_arm_joint_pos in __init__) and, on a misgrasp retry,
    # the target the arm is actually DRIVEN back to (PHASE_OBSERVE, real IK
    # motion, since that happens mid-trial and should look like genuine robot
    # behavior rather than a teleport).
    #
    # 0.40m + the original 24mm-focal-length lens only covered ~0.35m across (do
    # the math: 2*h*tan(atan((aperture/2)/focal))) -- nowhere near the 0.85m
    # table, which is exactly the "too close, can't see the whole table" you saw.
    # Simply raising the height further isn't safe either: at the height needed
    # to cover the table with THAT lens, the vantage point is ~1.17m from the
    # robot base -- past the Panda's ~0.85m reach, so the arm would never
    # actually get there. 0.55m + a 12mm lens (see grasp_camera.spawn below)
    # covers ~0.96m (13% margin over the table) at a reach distance of ~0.73m,
    # comfortably within range.
    observe_height: float = 0.55          # meters above the table surface, at table center
    observe_settle_steps: int = 100       # physics steps allowed to converge there (initial calibration)
    observe_timeout_steps: int = 150      # control steps allowed for PHASE_OBSERVE (retry case)

    # ── Grasp state-machine geometry ───────────────────────────────────────
    # Fallback grasp point when GG-CNN is unavailable/disabled/low-confidence:
    # the object's own root position (privileged, from the sim) with a fixed
    # top-down orientation -- matches how the ContactileHand env treats object_pos
    # as the reward-relevant point rather than a precisely computed grasp point.
    # When GG-CNN IS available, its predicted position/angle/width are used
    # instead (see ggcnn_* below) -- this offset still applies on top of it.
    pregrasp_hover_height: float = 0.15   # height above object to stage before descending
    # Z-offset applied to wherever the gripper closes -- for a naive fallback
    # grasp that's the object's root; for a GG-CNN/masked grasp, this is
    # actually the camera-visible TOP SURFACE point at the chosen pixel
    # (distance_to_image_plane at that pixel, unprojected), NOT the object's
    # root or center. At 0.0 (the old default), the gripper closes exactly at
    # that surface height -- fingers only ever reach the very top of the
    # object rather than descending far enough to wrap around its body, which
    # gives a shallow, easy-to-lose grip. Negative pushes the close point
    # DOWN (deeper into the object) so the fingers actually enclose some of
    # its body before closing. -0.015m is a starting point for small
    # (few-cm) objects -- tune per object size, and don't push deeper than
    # roughly the object's own height or the fingers will jam against the
    # table instead of the object.
    grasp_height_offset: float = -0.015
    lift_height: float = 0.30             # BenchmarkEnvCfg.max_lift_height is 0.20 --
                                           # bumped higher here on request; drop back
                                           # to 0.20 for an exact reference match
    # Matches heuristic_benchmark_env.py's threshold (0.6), not the PPO benchmark's
    # stricter 0.95 -- this is a scripted controller, not a trained policy, so the
    # heuristic env is the closer reference.
    success_lift_fraction: float = 0.6

    # ── Wrist-mounted (eye-in-hand) grasp camera + GG-CNN grasp prediction ──
    # Requires --enable_cameras. Attached to the hand link itself, NOT fixed in
    # the world -- a static table-mounted camera and a reaching arm inevitably
    # fight over the same airspace above the table; a wrist camera sidesteps that
    # entirely since there's no separate camera for the arm to block.
    #
    # Offset is in the panda_hand link's own LOCAL frame: pos is a small step back
    # from the hand origin along its local -Z (away from the fingers, which sit at
    # +tcp_offset_z -- see that field below), giving a bit of standoff so the
    # fingers themselves aren't right at the edge of frame. rot=(0,1,0,0) makes
    # the camera look along the hand's local +Z (the SAME direction the gripper
    # approaches from) instead of -Z -- reusing the identical 180°-about-X flip
    # already used for _down_quat_w elsewhere in this file, for the same reason
    # (this codebase's asset/convention needs that flip to point "the way the
    # gripper is pointing" rather than away from it). VERIFY this visually once
    # you can run it -- wrist-camera mounting offsets are exactly the kind of
    # thing that's easy to get backwards without seeing the actual render.
    grasp_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.06),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="opengl",
        ),
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # Confirmed-working values from actual runs: pos flipped to +0.06 (the
            # -0.06 guess had the offset direction backwards) and focal_length
            # 18mm (in between the original 24mm, too narrow, and this file's
            # earlier 12mm guess).
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
        width=480,
        height=480,
    )

    # No weights are shipped with this repo -- point this at your own trained
    # checkpoint (a torch state_dict matching ggcnn_model.GGCNN's layers). Empty
    # string disables GG-CNN entirely and falls back to the naive grasp above.
    ggcnn_checkpoint: str = ""
    ggcnn_input_size: int = 300           # network was trained at 300x300
    # GG-CNN's width output is a normalized 0-1 value calibrated to the ORIGINAL
    # dougsm/ggcnn repo's Cornell-dataset camera (fixed focal length + typical
    # working depth) -- that absolute-scale relationship does not transfer to a
    # different camera/depth setup without retraining, and empirically produced
    # widths ~5x too large here (0.19-0.20m raw vs an 0.08m max gripper opening,
    # consistently, across different grasps on the same object). Since this
    # benchmark already uses privileged 3D geometry for target identification
    # (_project_object_privileged_box), the actual commanded gripper width now
    # comes from the object's own known world-space footprint instead
    # (_get_object_world_footprint_extent) -- see grasp_width_margin below.
    # ggcnn_width_px_scale is kept only so the raw GG-CNN width can still be
    # logged for comparison; it no longer affects the commanded width.
    ggcnn_width_px_scale: float = 150.0
    # Multiplier applied to the target's true (narrower) horizontal footprint
    # extent to get the commanded gripper width -- >1.0 leaves finger clearance
    # around the object, <1.0 would try to squeeze inside it. 1.15 is a modest
    # 15% clearance; tune based on how snug/loose the grasps look in the debug
    # images (yellow/magenta jaw rectangle vs the actual object silhouette).
    # If True, always command the gripper to gripper_closed_pos (fully closed)
    # rather than computing an object-specific width -- a PD-controlled
    # parallel-jaw gripper naturally stalls against the object on contact
    # rather than crushing through it, so for objects well within the max
    # opening this is simpler and avoids needing ANY width estimate (learned
    # or privileged) at all. Set False to use the privileged-footprint width
    # computation instead (see grasp_width_margin below).
    close_gripper_fully: bool = True
    grasp_width_margin: float = 1.15
    # Below this predicted quality, distrust the network output and fall back to
    # the naive object-root grasp instead (e.g. target too occluded/ambiguous).
    ggcnn_min_quality: float = 0.15
    # No segmentation is used to isolate the target -- instead, GG-CNN is run over
    # the WHOLE scene, its top-K local-maximum quality candidates are each
    # unprojected to a 3D world point, and whichever candidate lands closest to the
    # target object's own (privileged) coordinate is picked. This is what actually
    # enforces "only grasp the target": a candidate sitting on a clutter object
    # will be far from the target's true coordinate and lose out to one that
    # isn't, even without ever segmenting pixels by instance.
    # NOTE: with only the top 10 GLOBAL quality peaks, a cluttered scene where the
    # target isn't among the highest-quality-looking objects can easily have ZERO
    # in-box candidates even though GG-CNN's full heatmap has a perfectly good
    # point sitting right on the target -- it just wasn't in the global top-10.
    # Raised for more headroom; the properly-fixed version searches the target's
    # box region directly on the full heatmap rather than hoping enough of the
    # global top-K happen to land there (needs ggcnn_model.py's raw output maps).
    ggcnn_top_k: int = 60
    # If even the best candidate is farther than this from the target's true
    # coordinate, treat GG-CNN as having failed to find the target at all and
    # fall back to the naive object-root grasp instead of grasping the wrong spot.
    ggcnn_max_candidate_dist: float = 0.12
    ggcnn_device: str = "cuda"
    # Debug tooling for actually seeing what GG-CNN did, not just trusting it.
    # Saves a PNG per capture (contrast-stretched depth, all candidates marked,
    # the chosen one highlighted) to ggcnn_debug_dir. Falls back to saving a raw
    # .npy array if Pillow isn't installed.
    ggcnn_debug_save_images: bool = True
    # Only actually WRITE debug images for every Nth trial (1 = every trial,
    # current behavior; 20 = roughly 5% of trials get images). Doesn't affect
    # console logging or trial_results.json, which still cover every trial --
    # this only controls the (expensive, disk-heavy) PNG writes. Essential for
    # large batch runs: at up to 20 attempts/trial x 2 images/attempt, a
    # 24-object x 4-mode x 100-trial sweep can otherwise generate hundreds of
    # thousands of PNGs and noticeably slow the run down with disk I/O.
    ggcnn_debug_save_trial_stride: int = 20
    ggcnn_debug_dir: str = "ggcnn_debug_images"

    # Save one RAW (undecorated -- no boxes, no candidate markers) snapshot of
    # the scene, using the wrist camera at its top-down observe pose, captured
    # ONCE for the entire run -- the first capture of trial 1 only, not every
    # trial. A "what did the very first layout look like" record, independent
    # of ggcnn_debug_* (which is for pipeline debugging, gated by the trial
    # stride, and always annotated).
    save_scene_images: bool = False
    scene_image_dir: str = "scene_images"
    # Below this depth standard deviation (meters), the capture is treated as
    # degenerate (flat/blank -- almost always a camera mounting/orientation
    # problem, not a GG-CNN problem) and skipped in favor of the naive fallback
    # rather than feeding GG-CNN garbage and trusting whatever it outputs.
    ggcnn_debug_min_depth_std: float = 0.005
    # Debug-image depth->grayscale contrast range (meters), separate from the
    # camera's actual sim clipping_range. Previously the debug PNG normalized
    # using THIS capture's own min/max depth -- if the frame happened to also
    # see any far background/off-table area (e.g. min=0.53 max=1.39m in one
    # observed capture), the real objects' narrow ~0.5-0.6m depth band got
    # compressed into a tiny sliver near the dark end, making the whole image
    # look almost solid black; a tighter capture (e.g. min=0.53 max=0.61m)
    # would stretch to a nice full-contrast grey image instead. Same object,
    # wildly different-looking debug images, purely from normalization -- not
    # an actual capture problem. Clip to this fixed working-height band before
    # stretching to 0-255 so every debug image gets consistent, comparable
    # contrast regardless of what else the frame happened to catch.
    ggcnn_debug_depth_display_range: tuple = (0.35, 0.75)

    # ── SAM object localization (ported from visual_grasp_classifier.py) ─────
    # SAM (Segment Anything) is class-agnostic segmentation -- it proposes masks
    # for "things" without naming them, unlike a detector that outputs class
    # labels. Since this is simulation, identity doesn't need to come from
    # vision at all: the target's own (privileged) 3D position is projected
    # into the image and matched against whichever SAM-proposed box contains
    # (or is nearest) that point -- an automated version of the reference
    # script's manual "pick an object_id" step. SAM's actual value here is
    # tight LOCALIZATION: crop to the matched box and run GG-CNN on that
    # focused crop instead of the whole cluttered frame. Empty checkpoint path
    # disables SAM entirely and falls back to whole-frame candidate matching.
    sam_checkpoint: str = ""
    sam_model_type: str = "vit_b"          # must match the checkpoint (vit_b/vit_l/vit_h)
    sam_points_per_side: int = 32          # mask-generator sampling density (higher = slower, finds smaller objects)
    sam_min_area_frac: float = 0.002       # drop masks smaller than this fraction of the image (noise)
    sam_max_area_frac: float = 0.4         # drop masks larger than this fraction of the image (background/table)
    sam_crop_margin_px: int = 20           # tolerance: how far outside a box's edge a GG-CNN
                                            # candidate pixel can still land and count as "in" it
    sam_nearest_fallback_max_px: float = 60.0  # nearest-centroid box fallback is discarded (treated
                                            # as no match) beyond this pixel distance from target_px --
                                            # without a cap, this fallback would confidently return an
                                            # arbitrarily far, wrong box rather than admit no match

    # Top-down parallel-jaw grasping cannot recover from arbitrary roll/pitch the way
    # a multi-fingered hand can. Restricting to yaw-only orientation keeps this a fair,
    # comparable benchmark; set False to match the ContactileHand env's fully random
    # orientation (expect a much lower success rate -- that's a real, meaningful result
    # too, just a different question).
    randomize_object_yaw_only: bool = True
    randomize_object_orientation: bool = True

    # Matches BenchmarkEnv's object_fall_margin -- how far below the table surface
    # counts as "fell off" (checked continuously, not just at spawn).
    object_fall_margin: float = 0.05
    # If a grasp attempt reaches lift height but the *target* object didn't come up
    # with it (empty gripper, or a clutter object got grasped instead), release and
    # redescend for another attempt instead of failing the trial outright. Not in the
    # original BenchmarkEnvCfg by this name (that env's retry bookkeeping is tangled
    # up with its hand-flip/stuck-recovery logic) -- this is a fresh, simpler budget
    # for the same idea. NOTE: this counts RETRIES, not total attempts -- with the
    # first attempt included, max_regrasp_attempts=19 means 20 total tries per trial
    # before giving up.
    max_regrasp_attempts: int = 19

    # Per-phase timeouts (steps at the decimated control rate). These just catch a
    # single phase stalling forever (e.g. IK not converging) -- the real "give up on
    # this trial" decision is the 1000-step episode cap above, same as the
    # reference's max_trial_timesteps.
    approach_timeout_steps: int = 150
    descend_timeout_steps: int = 150
    grasp_close_steps: int = 40
    # LIFT is evaluated for success THE MOMENT it ends (reached or timed out) -- if
    # this timeout fires before the arm has actually converged on the full
    # lift_height climb (e.g. slowed by clutter collisions), the height it gets
    # checked against the threshold at is whatever partial climb it reached by
    # then, not the intended full height. Generous on purpose so successful trials
    # reflect a real, (near-)complete lift rather than "just barely cleared the
    # threshold before running out of time."
    lift_timeout_steps: int = 300
    # After a successful lift, hold perfectly still at height for this many steps --
    # a brief, still visible "it worked" pause -- then the trial ends immediately (no
    # scripted return-to-home motion; the reset that follows teleports the robot
    # straight back to its home joint pose). A failed/exhausted trial skips the
    # freeze and ends immediately too.
    freeze_steps: int = 20
    pose_reach_pos_tol: float = 0.01   # m
    pose_reach_rot_tol: float = 0.08   # rad

    # ── Object spawning (same semantics as BenchmarkEnvCfg) ────────────────
    target_object_id: str = ""
    # Isolated mode = target only, no clutter -- always placed at table center
    # (see _generate_isolated_position), same as every other spawn mode now.
    use_isolated_mode: bool = False

    use_clutter_based_spawn: bool = False
    target_complexity: str = "C1_medium"

    min_objects_to_spawn: int = 1
    max_objects_to_spawn: int = 1

    # Also used by clutter spawn (_generate_clutter_based_positions) to bound its
    # neighbor placement -- one shared "how much of the table is actually in play"
    # knob for both modes.
    spawn_area_margin: float = 0.27
    min_object_spacing: float = 0.06
    max_spawn_attempts: int = 10
    spawn_settling_steps: int = 100
    spawn_height_tolerance: float = 0.01
    # Also doubles as the "reachable zone" bound checked every step during a trial --
    # if the object drifts past this from table center, the attempt is aborted as
    # out-of-reach (see PandaIKBenchmarkEnv._pre_physics_step).
    max_spawn_distance_from_origin: float = 0.30
    randomize_spawn_positions: bool = True

    object_usd_dir: str = USD_DIR
    available_objects: list[ObjectSpawnInfo] = field(default_factory=list)

    # ── Trials ──────────────────────────────────────────────────────────────
    num_trials: int = 10
    scene_random_seed: int = 42

    # ── Debug ───────────────────────────────────────────────────────────────
    enable_debug_state_machine: bool = False