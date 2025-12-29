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
from dataclasses import dataclass, field
from typing import Optional
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class NewtonSimParams(SimParams):
    """Newton-specific simulation parameters."""

    # MuJoCo/Newton uses fixed-size buffers for constraints and contacts.
    # If these are too small, the solver can overflow and produce NaNs.
    # If these are too large, Warp may OOM during initialization.
    #
    # Set `njmax` / `nconmax` directly to fully control sizing.
    # Otherwise, they are computed as max(base, per_env * num_envs).
    njmax: Optional[int] = None
    nconmax: Optional[int] = None
    njmax_per_env: int = 150
    nconmax_per_env: int = 40


@dataclass
class NewtonSimulatorConfig(SimulatorConfig):
    """Configuration specific to Newton simulator."""

    _target_: str = "protomotions.simulator.newton.simulator.NewtonSimulator"
    sim: NewtonSimParams = field(default_factory=NewtonSimParams)  # Override sim type
    w_last: bool = True  # Newton uses xyzw quaternions
