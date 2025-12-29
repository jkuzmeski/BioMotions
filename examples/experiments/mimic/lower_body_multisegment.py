# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
import argparse


"""
Lower Body Multi-Segment Mimic Environment Configuration
=========================================================

Motion tracking environment for the lower body multi-segment humanoid model
with detailed foot biomechanics (heel, medial toes, lateral metatarsal, lateral toes).

This configuration enables detailed ground reaction force analysis and 
realistic foot roll-off mechanics during gait tracking.

Key Features:
- 48 DOF (3 DOF per joint: hip, knee, ankle + 4 foot segments per leg)
- Contact sensors on all foot segments for GRF analysis
- Tuned rewards for biomechanically accurate motion tracking
"""


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    """Configure robot with contact sensors for detailed foot contact tracking.
    
    Enables contact sensing on all foot segments:
    - Heel: Posterior foot contact during heel strike
    - MedToes: Medial forefoot contact during toe-off
    - LatMeta: Lateral midfoot contact
    - LatToes: Lateral toes contact
    """
    robot_cfg.update_fields(
        contact_bodies=[
            # Left foot segments
            "L_Heel",
            "L_MedToes", 
            "L_LatMeta",
            "L_LatToes",
            # Right foot segments
            "R_Heel",
            "R_MedToes",
            "R_LatMeta", 
            "R_LatToes",
        ]
    )


def terrain_config(args: argparse.Namespace):
    """Build terrain configuration."""
    from protomotions.components.terrains.config import TerrainConfig

    return TerrainConfig()


def scene_lib_config(args: argparse.Namespace):
    """Build scene library configuration."""
    from protomotions.components.scene_lib import SceneLibConfig

    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    from protomotions.components.motion_lib import MotionLibConfig

    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> MimicEnvConfig:
    """Build environment configuration for lower body multi-segment model.
    
    Includes specialized rewards for:
    - Tracking accuracy on major leg joints
    - Foot segment pose and velocity tracking
    - Contact pattern matching for realistic gait
    """
    from protomotions.envs.mimic.config import (
        MimicEarlyTerminationEntry,
        MimicObsConfig,
        MimicMotionManagerConfig,
    )
    from protomotions.envs.obs.config import FuturePoseType, MimicTargetPoseConfig
    from protomotions.envs.base_env.config import RewardComponentConfig
    from protomotions.envs.obs.config import HumanoidObsConfig, ActionHistoryConfig
    from protomotions.envs.utils.rewards import (
        mean_squared_error_exp,
        rotation_error_exp,
        power_consumption_sum,
        norm,
        contact_mismatch_sum,
        impact_force_penalty,
    )

    mimic_early_termination = [
        MimicEarlyTerminationEntry(
            mimic_early_termination_key="max_joint_err",
            mimic_early_termination_thresh=0.5,
            less_than=False,
        )
    ]

    # Unified reward configuration
    reward_config = {
        # Base rewards
        "action_smoothness": RewardComponentConfig(
            function=norm,
            variables={
                "x": "current_actions - previous_actions",
            },
            weight=-0.02,
        ),
        # Mimic tracking rewards
        "gt_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos",
                "ref_x": "ref_state.rigid_body_pos",
                "coefficient": "-100.0",
            },
            weight=0.5,
        ),
        "gr_rew": RewardComponentConfig(
            function=rotation_error_exp,
            variables={
                "q": "current_state.rigid_body_rot",
                "ref_q": "ref_state.rigid_body_rot",
                "coefficient": "-5.0",
            },
            weight=0.3,
        ),
        "gv_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_vel",
                "ref_x": "ref_state.rigid_body_vel",
                "coefficient": "-0.5",
            },
            weight=0.1,
        ),
        "gav_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_ang_vel",
                "ref_x": "ref_state.rigid_body_ang_vel",
                "coefficient": "-0.1",
            },
            weight=0.1,
        ),
        "rh_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos[:, 0, 2]",  # Root height (z-coord of body 0)
                "ref_x": "ref_state.rigid_body_pos[:, 0, 2]",
                "coefficient": "-100.0",
            },
            weight=0.2,
        ),
        "pow_rew": RewardComponentConfig(
            function=power_consumption_sum,
            variables={
                "dof_forces": "current_state.dof_forces",
                "dof_vel": "current_state.dof_vel",
            },
            weight=-0.00005,
        ),
        # Contact tracking rewards for foot segments
        "contact_rew": RewardComponentConfig(
            function=contact_mismatch_sum,
            variables={
                "left_foot_contact": "current_state.contact_forces[:, :4, :]",  # Left foot 4 segments
                "right_foot_contact": "current_state.contact_forces[:, 4:, :]",  # Right foot 4 segments
                "ref_left_contact": "ref_state.contact_forces[:, :4, :]",
                "ref_right_contact": "ref_state.contact_forces[:, 4:, :]",
            },
            weight=-0.1,
        ),
    }

    return MimicEnvConfig(
        num_envs=4096,
        num_obs_steps=2,
        terminate_on_fall=True,
        motion_manager_config=MimicMotionManagerConfig(
            mimic_early_termination=mimic_early_termination,
        ),
        obs_config=MimicObsConfig(
            humanoid_obs_config=HumanoidObsConfig(
                enabled_obs=[
                    "joint_pos",
                    "joint_vel",
                    "angular_velocity",
                    "linear_velocity",
                    "local_root_rotation",
                    "height",
                ],
                action_history_config=ActionHistoryConfig(
                    include_action_history=True,
                    history_len=3,
                ),
            ),
            target_pose_config=MimicTargetPoseConfig(
                future_tracks=[
                    FuturePoseType(
                        steps=[5, 10, 15],  # Future frames to predict
                    )
                ],
                bodies_subset=[
                    "Pelvis",
                    "L_Hip",
                    "L_Knee", 
                    "L_Ankle",
                    "R_Hip",
                    "R_Knee",
                    "R_Ankle",
                ],  # Main tracking bodies
            ),
        ),
        reward_config=reward_config,
    )


def agent_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> PPOAgentConfig:
    """Build PPO agent configuration."""
    from protomotions.agents.ppo.config import PPOAlgoConfig, PPONetworkConfig

    return PPOAgentConfig(
        network=PPONetworkConfig(
            mlp_units=[1024, 512, 256],  # Larger network for 48 DOF
            activation="elu",
            separate_value_mlp=True,
        ),
        algorithm=PPOAlgoConfig(
            gamma=0.99,
            clip_param=0.2,
            mini_epochs=5,
            mini_batch_size=4096,
            learning_rate=5e-5,
            entropy_coef=0.01,
            value_loss_coef=0.5,
        ),
        max_epochs=5000,
    )
