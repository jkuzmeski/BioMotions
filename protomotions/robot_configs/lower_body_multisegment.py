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
"""
Lower Body Multi-Segment Robot Configuration.

This module provides configurations for the lower body multi-segment humanoid model
with detailed foot biomechanics including separate heel, medial toes, lateral metatarsal,
and lateral toes segments.

The model includes:
- Pelvis (root body with freejoint)
- L_Hip, L_Knee, L_Ankle (left leg chain)
- L_Heel, L_MedToes, L_LatMeta, L_LatToes (left foot segments)
- R_Hip, R_Knee, R_Ankle (right leg chain)
- R_Heel, R_MedToes, R_LatMeta, R_LatToes (right foot segments)

Each joint has 3 DOF (x, y, z axes) for a total of 48 DOF.
"""

from protomotions.robot_configs.base import (
    RobotAssetConfig,
    RobotConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimParams,
    IsaacLabPhysXParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# Default root height for the model (pelvis height when standing)
DEFAULT_ROOT_HEIGHT_M = 1.0

# PD gains for different joint groups
# Main leg joints - high stiffness for stability
HIP_STIFFNESS = 800.0
HIP_DAMPING = 80.0
KNEE_STIFFNESS = 800.0
KNEE_DAMPING = 80.0
ANKLE_STIFFNESS = 800.0
ANKLE_DAMPING = 80.0

# Foot segment joints - lower stiffness for compliance
HEEL_STIFFNESS = 500.0
HEEL_DAMPING = 50.0
MED_TOES_STIFFNESS = 500.0
MED_TOES_DAMPING = 50.0
LAT_META_STIFFNESS = 500.0
LAT_META_DAMPING = 50.0
LAT_TOES_STIFFNESS = 500.0
LAT_TOES_DAMPING = 50.0


@dataclass
class LowerBodyMultisegmentConfig(RobotConfig):
    """
    Configuration for lower body multi-segment humanoid model.
    
    This model features detailed foot biomechanics with separate segments for:
    - Heel: Posterior foot contact
    - MedToes: Medial forefoot and toes (hallux side)
    - LatMeta: Lateral metatarsals
    - LatToes: Lateral toes
    
    This configuration enables detailed ground reaction force analysis and
    realistic foot roll-off mechanics during gait.
    """
    
    # Bodies to track for motion capture and analysis
    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Pelvis",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "L_Heel",
            "L_MedToes",
            "L_LatMeta",
            "L_LatToes",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "R_Heel",
            "R_MedToes",
            "R_LatMeta",
            "R_LatToes",
        ]
    )
    
    # Bodies that can contact ground without terminating episode
    non_termination_contact_bodies: List[str] = field(
        default_factory=lambda: [
            # Left foot segments
            "L_Ankle",
            "L_Heel",
            "L_MedToes",
            "L_LatMeta",
            "L_LatToes",
            # Right foot segments
            "R_Ankle",
            "R_Heel",
            "R_MedToes",
            "R_LatMeta",
            "R_LatToes",
        ]
    )

    # Bodies with contact sensors for GRF analysis
    contact_bodies: List[str] = field(
        default_factory=lambda: [
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

    # IMU sensor body
    imu_body: Optional[str] = "Pelvis"

    # Common naming conventions for compatibility with other modules
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": [
                "L_Ankle",
                "L_Heel",
                "L_MedToes",
                "L_LatMeta",
                "L_LatToes",
            ],
            "all_right_foot_bodies": [
                "R_Ankle",
                "R_Heel",
                "R_MedToes",
                "R_LatMeta",
                "R_LatToes",
            ],
            "all_left_hand_bodies": [],  # No hands in lower body model
            "all_right_hand_bodies": [],
            "head_body_name": [],  # No head in lower body model
            "torso_body_name": ["Pelvis"],  # Pelvis acts as torso
        }
    )
    
    # Default root height for standing pose
    default_root_height: float = DEFAULT_ROOT_HEIGHT_M

    # Asset configuration
    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="protomotions/data/assets",
            asset_file_name="mjcf/lower_body_multisegment.xml",
            usd_asset_file_name="usd/lower_body_multisegment.usda",
            # Pelvis is the root body in the USD hierarchy - all bodies are children of Pelvis
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Pelvis/",
            self_collisions=False,  # Contact exclusions are defined in MJCF
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    # Control configuration with joint-specific PD gains
    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.TORQUE,
            action_scale=500.0,
            override_control_info={
                # Hip joints - highest stiffness for stability
                "L_Hip_.*": ControlInfo(
                    stiffness=HIP_STIFFNESS,
                    damping=HIP_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                "R_Hip_.*": ControlInfo(
                    stiffness=HIP_STIFFNESS,
                    damping=HIP_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                # Knee joints
                "L_Knee_.*": ControlInfo(
                    stiffness=KNEE_STIFFNESS,
                    damping=KNEE_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                "R_Knee_.*": ControlInfo(
                    stiffness=KNEE_STIFFNESS,
                    damping=KNEE_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                # Ankle joints
                "L_Ankle_.*": ControlInfo(
                    stiffness=ANKLE_STIFFNESS,
                    damping=ANKLE_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                "R_Ankle_.*": ControlInfo(
                    stiffness=ANKLE_STIFFNESS,
                    damping=ANKLE_DAMPING,
                    effort_limit=500,
                    velocity_limit=100,
                    armature=0.02,
                ),
                # Heel joints - moderate stiffness
                "L_Heel_.*": ControlInfo(
                    stiffness=HEEL_STIFFNESS,
                    damping=HEEL_DAMPING,
                    effort_limit=200,
                    velocity_limit=100,
                    armature=0.01,
                ),
                "R_Heel_.*": ControlInfo(
                    stiffness=HEEL_STIFFNESS,
                    damping=HEEL_DAMPING,
                    effort_limit=200,
                    velocity_limit=100,
                    armature=0.01,
                ),
                # Medial toes joints
                "L_MedToes_.*": ControlInfo(
                    stiffness=MED_TOES_STIFFNESS,
                    damping=MED_TOES_DAMPING,
                    effort_limit=100,
                    velocity_limit=100,
                    armature=0.01,
                ),
                "R_MedToes_.*": ControlInfo(
                    stiffness=MED_TOES_STIFFNESS,
                    damping=MED_TOES_DAMPING,
                    effort_limit=100,
                    velocity_limit=100,
                    armature=0.01,
                ),
                # Lateral metatarsal joints
                "L_LatMeta_.*": ControlInfo(
                    stiffness=LAT_META_STIFFNESS,
                    damping=LAT_META_DAMPING,
                    effort_limit=100,
                    velocity_limit=100,
                    armature=0.01,
                ),
                "R_LatMeta_.*": ControlInfo(
                    stiffness=LAT_META_STIFFNESS,
                    damping=LAT_META_DAMPING,
                    effort_limit=100,
                    velocity_limit=100,
                    armature=0.01,
                ),
                # Lateral toes joints - lowest stiffness
                "L_LatToes_.*": ControlInfo(
                    stiffness=LAT_TOES_STIFFNESS,
                    damping=LAT_TOES_DAMPING,
                    effort_limit=50,
                    velocity_limit=100,
                    armature=0.01,
                ),
                "R_LatToes_.*": ControlInfo(
                    stiffness=LAT_TOES_STIFFNESS,
                    damping=LAT_TOES_DAMPING,
                    effort_limit=50,
                    velocity_limit=100,
                    armature=0.01,
                ),
            },
        )
    )

    # Simulator-specific parameters
    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=1,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=8,
                    max_depenetration_velocity=2,
                ),
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=4,
            ),
        )
    )
