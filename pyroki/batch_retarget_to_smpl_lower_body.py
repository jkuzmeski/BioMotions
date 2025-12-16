#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""
Batch retarget keypoints to SMPL Lower Body humanoid using PyRoki.

Adapted for SMPL Lower Body (Pelvis + Legs)
"""

import time
from typing import Tuple, TypedDict
import glob
import os
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy

# Global configuration
SMPL_LINK_NAMES = None
N_RETARGET = 9  # 9 keypoints: Pelvis + 2*(Hip, Knee, Ankle, Toe)
N_AUX = 0       # No auxiliary points (hands/torso) for lower body

# Direct bone pairs for local alignment cost (Lower Body Only)
direct_pairs = [
    ("L_Hip", "L_Knee", 1.0),
    ("L_Knee", "L_Ankle", 1.0),
    ("L_Ankle", "L_Toe", 1.0),
    ("R_Hip", "R_Knee", 1.0),
    ("R_Knee", "R_Ankle", 1.0),
    ("R_Ankle", "R_Toe", 1.0),
]


def get_humanoid_retarget_indices() -> Tuple[list, jnp.ndarray]:
    """Map human keypoint names to SMPL URDF link indices."""
    human_retarget_names = []
    smpl_joint_retarget_indices = []

    # Map generic keypoint names to your SPECIFIC URDF link names
    for human_name, smpl_name in [
        ("Pelvis", "Pelvis"),
        ("L_Hip", "L_Hip"),
        ("L_Knee", "L_Knee"),
        ("L_Ankle", "L_Ankle"),
        ("L_Toe", "L_Toe"),
        ("R_Hip", "R_Hip"),
        ("R_Knee", "R_Knee"),
        ("R_Ankle", "R_Ankle"),
        ("R_Toe", "R_Toe"),
    ]:
        human_retarget_names.append(human_name)
        smpl_joint_retarget_indices.append(SMPL_LINK_NAMES.index(smpl_name))

    smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices)
    return human_retarget_names, smpl_joint_retarget_indices


human_retarget_names, smpl_joint_retarget_indices = None, None


def load_motion_data(motion_path, source_type, subsample_factor, target_raw_frames):
    """Load and process motion data from a keypoints file."""
    print(f"Loading motion from: {motion_path}")
    motion_data = onp.load(motion_path, allow_pickle=True).item()

    raw_positions = motion_data["positions"]
    raw_orientations = motion_data["orientations"]
    raw_left_foot_contacts = motion_data["left_foot_contacts"]
    raw_right_foot_contacts = motion_data["right_foot_contacts"]
    original_raw_frames = raw_positions.shape[0]

    if target_raw_frames <= 0:
        target_raw_frames = original_raw_frames

    target_subsampled_frames = len(list(range(0, target_raw_frames, subsample_factor)))

    print(f"Original motion length: {original_raw_frames} frames.")

    # Calculate display frames
    original_subsampled_display_count = raw_positions[::subsample_factor].shape[0]
    num_timesteps = min(original_subsampled_display_count, target_subsampled_frames)

    print(f"Motion will be displayed for {num_timesteps} subsampled frames.")

    # Pad or trim to target_raw_frames
    if original_raw_frames >= target_raw_frames:
        processed_positions = raw_positions[:target_raw_frames]
        processed_orientations = raw_orientations[:target_raw_frames]
        processed_left_contacts = raw_left_foot_contacts[:target_raw_frames]
        processed_right_contacts = raw_right_foot_contacts[:target_raw_frames]
    else:
        padding_count = target_raw_frames - original_raw_frames

        pos_padding = onp.repeat(raw_positions[-1:], padding_count, axis=0)
        processed_positions = onp.concatenate((raw_positions, pos_padding), axis=0)

        orient_padding = onp.repeat(raw_orientations[-1:], padding_count, axis=0)
        processed_orientations = onp.concatenate(
            (raw_orientations, orient_padding), axis=0
        )

        left_padding = onp.repeat(raw_left_foot_contacts[-1:], padding_count, axis=0)
        processed_left_contacts = onp.concatenate(
            (raw_left_foot_contacts, left_padding), axis=0
        )

        right_padding = onp.repeat(raw_right_foot_contacts[-1:], padding_count, axis=0)
        processed_right_contacts = onp.concatenate(
            (raw_right_foot_contacts, right_padding), axis=0
        )

    # Process contact labels - maximum of ankle and toe contacts (foot in contact if EITHER is touching)
    left_contacts_avg = onp.max(processed_left_contacts.astype(float), axis=1)[:, None]
    right_contacts_avg = onp.max(processed_right_contacts.astype(float), axis=1)[
        :, None
    ]

    # Apply cross-fade (sliding window average) for smooth transitions
    window_size = 5

    def apply_crossfade(contact_flags):
        smoothed = onp.zeros_like(contact_flags)
        for i in range(len(contact_flags)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(contact_flags), i + window_size // 2 + 1)
            smoothed[i] = onp.mean(contact_flags[start_idx:end_idx])
        return smoothed

    left_contacts_smoothed = apply_crossfade(left_contacts_avg)
    right_contacts_smoothed = apply_crossfade(right_contacts_avg)

    # Subsample for solver
    simplified_keypoints = processed_positions[::subsample_factor]

    # Scaling Logic (Adapted for SMPL Lower Body)
    if source_type == "treadmill":
        # Treadmill data usually 1:1
        pass
    elif source_type == "smpl":
        # 0.9 Scaling often used for SMPL -> Robot mapping
        keypoints_root = simplified_keypoints[:, 0, :]
        keypoints_local = simplified_keypoints - keypoints_root[:, None, :]
        
        # Scale lower body
        keypoints_local = keypoints_local * onp.array([0.9, 0.9, 0.85])[None, None, :]
        keypoints_root = keypoints_root * onp.array([0.9, 0.9, 0.85])[None, :]
        
        simplified_keypoints = keypoints_root[:, None, :] + keypoints_local
        
        # We only have 9 keypoints for lower body, ensure shape is correct
        if simplified_keypoints.shape[1] > N_RETARGET:
             simplified_keypoints = simplified_keypoints[:, :N_RETARGET, :]

    keypoint_orientations = processed_orientations[::subsample_factor]
    left_foot_contact = left_contacts_smoothed[::subsample_factor]
    right_foot_contact = right_contacts_smoothed[::subsample_factor]

    return (
        simplified_keypoints,
        keypoint_orientations,
        left_foot_contact,
        right_foot_contact,
        num_timesteps,
    )


def save_contact_labels(output_path, left_contact, right_contact, num_timesteps):
    """Save processed foot contact labels to disk."""
    left_contacts = left_contact[:num_timesteps].squeeze(-1)
    right_contacts = right_contact[:num_timesteps].squeeze(-1)
    foot_contacts = onp.stack([left_contacts, right_contacts], axis=-1)
    onp.savez_compressed(output_path, foot_contacts=foot_contacts)
    print(f"Saved contact labels to {output_path} with shape {foot_contacts.shape}")


class RetargetingWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    root_smoothness: float
    joint_smoothness: float
    self_collision: float
    joint_rest_penalty: float
    joint_vel_limit: float
    foot_contact: float
    foot_tilt: float


@jaxls.Cost.create_factory
def joint_vel_limit_cost(
    var_values: jaxls.VarValues,
    var_joints_curr: jaxls.Var[jnp.ndarray],
    var_joints_prev: jaxls.Var[jnp.ndarray],
    max_vel: float,
    dt: float,
    weight: float,
) -> jax.Array:
    joints_curr = var_values[var_joints_curr]
    joints_prev = var_values[var_joints_prev]
    joint_vel = (joints_curr - joints_prev) / dt
    excess_vel = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)
    return excess_vel.flatten() * weight


@jaxls.Cost.create_factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root_curr: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    smpl_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    """Penalize foot movement, height diff, and ground penetration when in contact."""
    T_world_root_curr = var_values[var_Ts_world_root_curr]
    T_world_root_prev = var_values[var_Ts_world_root_prev]
    robot_cfg_curr = var_values[var_robot_cfg_curr]
    robot_cfg_prev = var_values[var_robot_cfg_prev]

    T_root_link_curr = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_curr))
    T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_prev))
    T_world_link_curr = T_world_root_curr @ T_root_link_curr
    T_world_link_prev = T_world_root_prev @ T_root_link_prev

    left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx = foot_indices

    # Map to robot indices
    left_ankle_robot_idx = smpl_joint_retarget_indices[left_ankle_idx]
    right_ankle_robot_idx = smpl_joint_retarget_indices[right_ankle_idx]
    left_foot_robot_idx = smpl_joint_retarget_indices[left_foot_idx]
    right_foot_robot_idx = smpl_joint_retarget_indices[right_foot_idx]

    robot_positions_curr = T_world_link_curr.translation()
    robot_positions_prev = T_world_link_prev.translation()

    # Current positions
    left_ankle_curr = robot_positions_curr[left_ankle_robot_idx]
    right_ankle_curr = robot_positions_curr[right_ankle_robot_idx]
    left_foot_curr = robot_positions_curr[left_foot_robot_idx]
    right_foot_curr = robot_positions_curr[right_foot_robot_idx]

    # Previous positions (for velocity)
    left_ankle_prev = robot_positions_prev[left_ankle_robot_idx]
    right_ankle_prev = robot_positions_prev[right_ankle_robot_idx]
    left_foot_prev = robot_positions_prev[left_foot_robot_idx]
    right_foot_prev = robot_positions_prev[right_foot_robot_idx]

    # Velocities
    left_ankle_vel = left_ankle_curr - left_ankle_prev
    right_ankle_vel = right_ankle_curr - right_ankle_prev
    left_foot_vel = left_foot_curr - left_foot_prev
    right_foot_vel = right_foot_curr - right_foot_prev

    # Z-height consistency between ankle and toe
    left_z_diff = left_ankle_curr[2] - left_foot_curr[2]
    right_z_diff = right_ankle_curr[2] - right_foot_curr[2]

    # Ground contact constraint: penalize being ABOVE ground when in contact
    # Use minimum of ankle and toe z-positions as the lowest point of the foot
    left_min_z = jnp.minimum(left_ankle_curr[2], left_foot_curr[2])
    right_min_z = jnp.minimum(right_ankle_curr[2], right_foot_curr[2])
    
    # Penalize positive z (above ground) when in contact
    left_ground_penetration = jnp.maximum(left_min_z, 0.0)
    right_ground_penetration = jnp.maximum(right_min_z, 0.0)

    # Costs
    left_w = left_foot_contact[0]
    right_w = right_foot_contact[0]

    return (
        jnp.concatenate([
            left_w * left_ankle_vel,
            right_w * right_ankle_vel,
            left_w * left_foot_vel,
            right_w * right_foot_vel,
            jnp.array([left_w * left_z_diff]),
            jnp.array([right_w * right_z_diff]),
            jnp.array([left_w * left_ground_penetration * 5.0]),  # Strong penalty for lifting off ground
            jnp.array([right_w * right_ground_penetration * 5.0]),
        ]).flatten() * weight
    )


@jaxls.Cost.create_factory
def foot_tilt_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    smpl_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    """Cost to keep feet flat (Z-up) when in contact."""
    T_world_root = var_values[var_Ts_world_root]
    robot_cfg = var_values[var_robot_cfg]
    T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    T_world_link = T_world_root @ T_root_link

    left_ankle_idx, right_ankle_idx, _, _ = foot_indices
    left_idx = smpl_joint_retarget_indices[left_ankle_idx]
    right_idx = smpl_joint_retarget_indices[right_ankle_idx]

    left_ori = T_world_link.rotation().as_matrix()[left_idx]
    right_ori = T_world_link.rotation().as_matrix()[right_idx]

    # Z-axis (col 2) should be [0,0,1], so z-component (2,2) should be 1.0
    left_tilt = left_foot_contact[0] * (left_ori[2, 2] - 1.0)
    right_tilt = right_foot_contact[0] * (right_ori[2, 2] - 1.0)

    return jnp.array([left_tilt, right_tilt]) * weight


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision | None,
    target_keypoints: jnp.ndarray,
    target_orientations: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    smpl_joint_retarget_indices: jnp.ndarray,
    smpl_retarget_mask: jnp.ndarray,
    weights: RetargetingWeights,
    subsample_factor: int = 1,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the retargeting optimization problem."""
    n_retarget = len(smpl_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Foot indices for contact costs
    foot_indices = jnp.array([
        human_retarget_names.index("L_Ankle"),
        human_retarget_names.index("R_Ankle"),
        human_retarget_names.index("L_Toe"),
        human_retarget_names.index("R_Toe"),
    ])

    # Variables
    class SimplifiedJointsScaleVarSMPL(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_joints_scale = SimplifiedJointsScaleVarSMPL(jnp.zeros(timesteps))

    # Initialize root from source
    root_init_se3_list = []
    for t in range(timesteps):
        root_pos = target_keypoints[t, 0, :]
        
        # FIX: Force UPRIGHT initialization (Ignore input rotation to prevent "folded" bugs)
        root_se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), root_pos
        )
        root_init_se3_list.append(root_se3)

    root_init_values = jaxlie.SE3(
        jnp.stack([se3.wxyz_xyz for se3 in root_init_se3_list])
    )

    # --- Cost Functions ---

    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarSMPL,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Local alignment cost (Relative positions)."""
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        target_pos = keypoints[:N_RETARGET, :]
        robot_pos = T_world_link.translation()[jnp.array(smpl_joint_retarget_indices)]

        delta_target = target_pos[:, None] - target_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        position_scale = var_values[var_joints_scale][..., None]
        residual_position = (
            (delta_target - delta_robot * position_scale)
            * (1 - jnp.eye(delta_target.shape[0])[..., None])
            * smpl_retarget_mask[..., None]
        )

        delta_target_norm = delta_target / (jnp.linalg.norm(delta_target + 1e-6, axis=-1, keepdims=True))
        delta_robot_norm = delta_robot / (jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True))
        residual_angle = 1 - (delta_target_norm * delta_robot_norm).sum(axis=-1)
        residual_angle = residual_angle * (1 - jnp.eye(residual_angle.shape[0])) * smpl_retarget_mask

        return jnp.concatenate([
            residual_position.flatten(),
            residual_angle.flatten()
        ]) * weights["local_alignment"]

    @jaxls.Cost.create_factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_joints_scale: SimplifiedJointsScaleVarSMPL,
    ) -> jax.Array:
        res_0 = (var_values[var_joints_scale] - 1.0).flatten() * 1.0
        res_1 = (var_values[var_joints_scale] - var_values[var_joints_scale].T).flatten() * 100.0
        res_2 = jnp.clip(-var_values[var_joints_scale], min=0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    @jaxls.Cost.create_factory
    def global_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarSMPL,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Global position alignment cost."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[smpl_joint_retarget_indices]

        keypoint_pos = keypoints[:N_RETARGET, :]

        return (link_pos - keypoint_pos).flatten() * weights["global_alignment"]

    @jaxls.Cost.create_factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        return (
            var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
        ).log().flatten() * weights["root_smoothness"]

    # --- Build Problem ---
    costs = [
        retargeting_cost(var_Ts_world_root, var_joints, var_joints_scale, target_keypoints),
        scale_regularization(var_joints_scale),
        pk.costs.limit_cost(jax.tree.map(lambda x: x[None], robot), var_joints, 100.0),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            weights["joint_smoothness"],
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        global_alignment_cost(var_Ts_world_root, var_joints, var_joints_scale, target_keypoints),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(var_joints.default_factory().shape, weights["joint_rest_penalty"])[None],
        ),
        joint_vel_limit_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            20.0,
            subsample_factor / 30.0,
            weights["joint_vel_limit"],
        ),
    ]

    # Add per-timestep costs
    for t in range(1, timesteps):
        costs.append(foot_contact_cost(
            jaxls.SE3Var(t), jaxls.SE3Var(t - 1),
            robot.joint_var_cls(t), robot.joint_var_cls(t - 1),
            robot,
            left_foot_contact[t], right_foot_contact[t],
            smpl_joint_retarget_indices, foot_indices,
            weights["foot_contact"],
        ))

    for t in range(timesteps):
        costs.append(foot_tilt_cost(
            jaxls.SE3Var(t), robot.joint_var_cls(t),
            robot,
            left_foot_contact[t], right_foot_contact[t],
            smpl_joint_retarget_indices, foot_indices,
            weights["foot_tilt"],
        ))

    # Solve
    solution = (
        jaxls.LeastSquaresProblem(costs, [var_joints, var_Ts_world_root, var_joints_scale])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([
                var_joints,
                var_Ts_world_root.with_value(root_init_values),
                var_joints_scale,
            ]),
            termination=jaxls.TerminationConfig(max_iterations=800),
        )
    )

    return solution[var_Ts_world_root], solution[var_joints]


def main():
    SCRIPT_DIR = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="SMPL Lower Body Retargeting")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize")
    parser.add_argument("--keypoints-folder-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./retargeted_output_motions")
    parser.add_argument("--urdf-path", type=str, default=str(SCRIPT_DIR / "../protomotions/data/assets/urdf/for_retargeting/smpl_lower_body.urdf"))
    parser.add_argument("--mesh-dir", type=str, default=str(SCRIPT_DIR / "../protomotions/data/assets/mesh"))
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument("--target-raw-frames", type=int, default=450)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--source-type", type=str, default="treadmill")
    parser.add_argument("--save-contacts-only", action="store_true")
    parser.add_argument("--contacts-dir", type=str, default=None)

    args = parser.parse_args()

    # Find files
    keypoints_paths = sorted(glob.glob(os.path.join(args.keypoints_folder_path, "*.npy")))
    if not keypoints_paths:
        print(f"No .npy files found in {args.keypoints_folder_path}")
        return

    # Check for Contact-Only Mode
    if args.save_contacts_only:
        print("Running in save-contacts-only mode.")
        contacts_dir = args.contacts_dir or os.path.join(args.keypoints_folder_path, "contacts")
        os.makedirs(contacts_dir, exist_ok=True)
        for i, motion_path in enumerate(keypoints_paths):
            print(f"Processing contacts {i+1}/{len(keypoints_paths)}")
            base_filename = os.path.splitext(os.path.basename(motion_path))[0]
            output_path = os.path.join(contacts_dir, f"{base_filename}_contacts.npz")
            if args.skip_existing and os.path.exists(output_path): continue
            
            _, _, left, right, steps = load_motion_data(
                motion_path, args.source_type, args.subsample_factor, args.target_raw_frames
            )
            save_contact_labels(output_path, left, right, steps)
        return

    # Init Robot
    global SMPL_LINK_NAMES, human_retarget_names, smpl_joint_retarget_indices
    urdf = yourdfpy.URDF.load(args.urdf_path, mesh_dir=args.mesh_dir)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = None
    if SMPL_LINK_NAMES is None: SMPL_LINK_NAMES = list(robot.links.names)
    human_retarget_names, smpl_joint_retarget_indices = get_humanoid_retarget_indices()

    # Get joint names for saving
    joint_names = getattr(robot, "joint_names", [])
    if not joint_names and hasattr(robot, "dof_names"):
        joint_names = robot.dof_names
    
    # Fallback if still empty: try to get from URDF actuated joints
    if not joint_names:
        print("Warning: Could not get joint names from robot. Trying URDF actuated joints...")
        # Note: This order might not match PyRoki's internal order if it reorders them.
        # But usually PyRoki follows URDF order.
        joint_names = urdf.actuated_joint_names
    
    print(f"Detected {len(joint_names)} joints: {joint_names}")

    # Create Mask
    n_retarget = len(smpl_joint_retarget_indices)
    smpl_retarget_mask = jnp.zeros((n_retarget, n_retarget))
    for link_a, link_b, weight in direct_pairs:
        idx_a = human_retarget_names.index(link_a)
        idx_b = human_retarget_names.index(link_b)
        smpl_retarget_mask = smpl_retarget_mask.at[idx_a, idx_b].set(weight)
        smpl_retarget_mask = smpl_retarget_mask.at[idx_b, idx_a].set(weight)

    # Weights
    weights_dict = RetargetingWeights(
        local_alignment=10.0, global_alignment=20.0, root_smoothness=5.0,
        joint_smoothness=10.0, self_collision=0.0, joint_rest_penalty=0.5,
        joint_vel_limit=50.0, foot_contact=30.0, foot_tilt=5.0,
    )

    if not args.visualize:
        print("Running batch retargeting...")
        os.makedirs(args.output_dir, exist_ok=True)
        for i, motion_path in enumerate(keypoints_paths):
            print(f"Processing {i+1}/{len(keypoints_paths)}: {os.path.basename(motion_path)}")
            base = os.path.splitext(os.path.basename(motion_path))[0]
            out = os.path.join(args.output_dir, f"{base}_retargeted.npz")
            if args.skip_existing and os.path.exists(out): continue

            keys, orients, left, right, steps = load_motion_data(
                motion_path, args.source_type, args.subsample_factor, args.target_raw_frames
            )
            root, joints = solve_retargeting(
                robot, robot_coll, keys, orients, left, right,
                smpl_joint_retarget_indices, smpl_retarget_mask, weights_dict, args.subsample_factor
            )
            onp.savez_compressed(out,
                base_frame_pos=onp.array(root.wxyz_xyz[:steps, 4:]),
                base_frame_wxyz=onp.array(root.wxyz_xyz[:steps, :4]),
                joint_angles=onp.array(joints[:steps]),
                joint_names=onp.array(joint_names)
            )
            print(f"Saved: {out}")
    else:
        # VISUALIZATION MODE (Windows Friendly)
        import viser
        from viser.extras import ViserUrdf
        
        curr_idx = 0
        keys, orients, left, right, steps = load_motion_data(
            keypoints_paths[curr_idx], args.source_type, args.subsample_factor, args.target_raw_frames
        )

        print("[INFO] Starting Viser Server...")
        server = viser.ViserServer(port=8081)
        print("\n" + "="*60)
        print(" VISUALIZER READY!")
        print(" OPEN THIS IN BROWSER: http://localhost:8081")
        print("="*60 + "\n")

        base_frame = server.scene.add_frame("/base", show_axes=False)
        server.scene.add_grid("ground", width=20, height=20, cell_size=1.0, plane="xy")
        urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
        playing = server.gui.add_checkbox("playing", True)
        slider = server.gui.add_slider("timestep", 0, 100, 1, 0)
        slider.max = steps - 1 if steps > 0 else 0

        server.gui.add_button("Reset Timeline").on_click(lambda _: setattr(slider, 'value', 0))
        weights_tuner = pk.viewer.WeightTuner(server, weights_dict) # type: ignore

        Ts_world_root, joints = None, None

        def generate_trajectory():
            nonlocal Ts_world_root, joints
            print(f"Retargeting {os.path.basename(keypoints_paths[curr_idx])}...")
            btn_retarget.disabled = True
            btn_next.disabled = True
            
            Ts_world_root, joints = solve_retargeting(
                robot, robot_coll, keys, orients, left, right,
                smpl_joint_retarget_indices, smpl_retarget_mask, 
                weights_tuner.get_weights(), # type: ignore
                args.subsample_factor
            )
            btn_retarget.disabled = False
            btn_next.disabled = False
            print("Done!")

        def next_motion(_):
            nonlocal curr_idx, keys, orients, left, right, steps
            curr_idx = (curr_idx + 1) % len(keypoints_paths)
            print(f"Next: {os.path.basename(keypoints_paths[curr_idx])}")
            keys, orients, left, right, steps = load_motion_data(
                keypoints_paths[curr_idx], args.source_type, args.subsample_factor, args.target_raw_frames
            )
            slider.max = steps - 1 if steps > 0 else 0
            slider.value = 0
            generate_trajectory()

        def save_motion(_):
            if Ts_world_root is None:
                print("No motion to save!")
                return
            
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(keypoints_paths[curr_idx]))[0]
            out = os.path.join(args.output_dir, f"{base}_retargeted.npz")
            
            # Try to get joint names to ensure correct ordering
            joint_names = getattr(robot, "joint_names", [])
            if not joint_names and hasattr(robot, "dof_names"):
                joint_names = robot.dof_names
            
            print(f"Saving with {len(joint_names)} joint names: {joint_names}")

            onp.savez_compressed(out,
                base_frame_pos=onp.array(Ts_world_root.wxyz_xyz[:steps, 4:]),
                base_frame_wxyz=onp.array(Ts_world_root.wxyz_xyz[:steps, :4]),
                joint_angles=onp.array(joints[:steps]),
                joint_names=onp.array(joint_names)
            )
            print(f"Saved: {out}")

        btn_retarget = server.gui.add_button("Retarget!")
        btn_retarget.on_click(lambda _: generate_trajectory())
        btn_save = server.gui.add_button("Save Motion")
        btn_save.on_click(save_motion)
        btn_next = server.gui.add_button("Next Motion")
        btn_next.on_click(next_motion)

        generate_trajectory()

        while True:
            with server.atomic():
                if playing.value and steps > 0:
                    slider.value = (slider.value + 1) % steps
                tstep = slider.value

            try:
                if Ts_world_root is not None:
                    base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
                    base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])
                    urdf_vis.update_cfg(onp.array(joints[tstep]))
                
                server.scene.add_point_cloud(
                    "/target_keypoints",
                    onp.array(keys[tstep]),
                    onp.array((0, 0, 255))[None].repeat(keys.shape[1], axis=0),
                    point_size=0.01,
                )
            except: pass
            time.sleep(0.03 * args.subsample_factor)

if __name__ == "__main__":
    main()