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

# Workaround for MuJoCo DLL loading issue on Windows when using Isaac Lab.
# MuJoCo's bundled plugins cause DLL conflicts with Isaac Lab's OpenGL context.
# We patch ctypes.CDLL to skip loading problematic DLLs before importing mujoco.
# See: https://github.com/google-deepmind/mujoco/issues/1164
import os
import sys
if sys.platform == "win32":
    import ctypes
    _original_cdll_init = ctypes.CDLL.__init__

    def _patched_cdll_init(self, name, *args, **kwargs):
        # Skip loading MuJoCo plugin DLLs that cause conflicts
        if name and "mujoco" in name.lower() and "plugin" in name.lower():
            self._handle = None
            self._name = name
            return
        return _original_cdll_init(self, name, *args, **kwargs)

    ctypes.CDLL.__init__ = _patched_cdll_init

"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation environment
for inference, visualization, and analysis. It supports interactive controls,
video recording, and motion playback.

Motion Playback
---------------

For kinematic motion playback (no physics simulation)::

    PYTHON_PATH protomotions/inference_agent.py \\
        --config-name play_motion \\
        +robot=smpl \\
        +simulator=isaacgym \\
        +motion_file=data/motions/walk.motion

Inference Config System
------------------------

Inference loads frozen configs from resolved_configs_inference.pt and applies inference-specific overrides.

Override Priority:

1. CLI overrides (--overrides) - Highest (runtime control)
2. Experiment inference overrides (apply_inference_overrides) - High (experiment-specific inference settings)
3. Frozen configs from resolved_configs.pt - Lowest (exact training configs)

Note: configure_robot_and_simulator() is NOT called during inference (already baked into frozen configs).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **L**: Start/stop video recording
- **Q**: Quit

Example
-------
>>> # Test with custom settings
>>> # PYTHON_PATH protomotions/inference_agent.py \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     +checkpoint=results/tracker/last.ckpt \\
>>> #     motion_file=data/motions/test.pt \\
>>> #     num_envs=16
"""


def create_parser():
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="default",
        help="Name of the experiment subfolder for saving biomechanics data",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import torch  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


# def tmp_enable_domain_randomization(robot_cfg, simulator_cfg, env_cfg):
#     """Temporary function to enable domain randomization for testing.

#     TODO: find a better way for sophisticated tmp inference overrides beyond CLI.
#     """
#     from protomotions.simulator.base_simulator.config import (
#         # FrictionDomainRandomizationConfig,
#         CenterOfMassDomainRandomizationConfig,
#         DomainRandomizationConfig,
#     )

#     # env_cfg.terrain.sim_config.static_friction = 0.01
#     # env_cfg.terrain.sim_config.dynamic_friction = 0.01

#     simulator_cfg.domain_randomization = DomainRandomizationConfig(
#         # Uncomment to enable action noise and friction randomization:
#         # action_noise=ActionNoiseDomainRandomizationConfig(
#         #     action_noise_range=(-0.01, 0.01),
#         #     dof_names=[".*"],
#         #     dof_indices=None
#         # ),
#         # friction=FrictionDomainRandomizationConfig(
#         #     num_buckets=64,
#         #     static_friction_range=(0.0, 1.0),
#         #     dynamic_friction_range=(0.0, 1.0),
#         #     restitution_range=(0.0, 0.0),
#         #     body_names=[".*"],
#         #     body_indices=None
#         # ),
#     )
#     log.info("Enabled domain randomization for testing")


def main():
    # Re-use the parser and args from module level
    global parser, args
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    # Load frozen configs from resolved_configs.pt (exact reproducibility)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Check if we need to switch simulators
    # Extract simulator name from current config's _target_
    current_simulator = simulator_config._target_.split(
        "."
    )[
        -3
    ]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )
    # Apply backward compatibility fixes for old checkpoints
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # # Temporary: Enable domain randomization for testing (uncomment to use)
    # tmp_enable_domain_randomization(robot_config, simulator_config, env_config)

    # Apply CLI runtime overrides
    if args.num_envs is not None:
        log.info(f"CLI override: num_envs = {args.num_envs}")
        simulator_config.num_envs = args.num_envs

    if args.motion_file is not None:
        log.info(f"CLI override: motion_file = {args.motion_file}")
        motion_lib_config.motion_file = args.motion_file  # Always present

    if args.scenes_file is not None:
        log.info(f"CLI override: scenes_file = {args.scenes_file}")
        scene_lib_config.scene_file = args.scenes_file  # Always present

    if args.headless is not None:
        log.info(f"CLI override: headless = {args.headless}")
        simulator_config.headless = args.headless

    # Parse and apply general CLI overrides
    from protomotions.utils.config_utils import (
        parse_cli_overrides,
        apply_config_overrides,
    )

    cli_overrides = parse_cli_overrides(args.overrides) if args.overrides else None

    if cli_overrides:
        apply_config_overrides(
            cli_overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    # Create fabric config for inference (single device, no distributed training)
    from lightning.fabric.strategies import SingleDeviceStrategy
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fabric_config = FabricConfig(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_nodes=1,
        strategy=SingleDeviceStrategy(device=device),
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**fabric_config.to_dict())
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    # Create components (terrain, scene_lib, motion_lib, simulator)
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create env (auto-initializes simulator)
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = checkpoint_path.parent

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    # agent_config.evaluator.eval_metric_keys = [
    #     "gt_err",
    #     "gr_err_degrees",
    #     "pow_rew",
    #     "gt_left_foot_contact",
    #     "gt_right_foot_contact",
    #     "pred_left_foot_contact",
    #     "pred_right_foot_contact"
    # ]
    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )

    agent.setup()
    agent.load(args.checkpoint, load_env=False)

    if args.full_eval:
        agent.evaluator.eval_count = 0
        agent.evaluator.evaluate()
    else:
        # agent.evaluator.simple_test_policy(collect_metrics=True)
        # Custom biomechanics inference loop
        log.info(f"Starting biomechanics inference for experiment: {args.experiment_name}")
        
        # Setup output directory
        output_dir = checkpoint.parent / "biomechanics" / args.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving biomechanics data to: {output_dir}")

        agent.eval()
        
        # Data storage
        collected_data = {
            "dof_pos": [],
            "dof_vel": [],
            "dof_forces": [],  # Joint torques
            "rigid_body_contact_forces": [],  # GRF (if sensors on feet)
            "rigid_body_pos": [],  # Needed for COP computation
            "root_pos": [],
            "root_rot": [],
            "root_vel": [],
            "root_ang_vel": [],
            "imu_orientation": [],
            "imu_ang_vel": [],
            "imu_lin_acc": [],
            "rewards": [],
            "dones": []
        }

        # Simulation loop
        # We run for a fixed duration or until all envs are done once? 
        # For biomechanics analysis, usually a fixed duration is good.
        # Let's run for motion length if available, or a fixed number of steps.
        # Assuming the motion is cyclic or we want to capture steady state.
        
        # Determine max steps: use motion length if available (conceptually), 
        # but here we might just run for env.max_episode_length
        max_steps = env.max_episode_length if hasattr(env, "max_episode_length") else 1000
        log.info(f"Running inference for {max_steps} steps...")

        with torch.no_grad():
            obs, _ = env.reset()
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)
            
            from rich.progress import track
            for step in track(range(max_steps), description="Simulating..."):
                # 1. Get Robot State from Simulator
                # env.simulator.get_robot_state() returns a RobotState with all data
                robot_state = env.simulator.get_robot_state()
                
                # 2. Store Data (move to CPU to save memory)
                collected_data["dof_pos"].append(robot_state.dof_pos.cpu())
                collected_data["dof_vel"].append(robot_state.dof_vel.cpu())
                collected_data["dof_forces"].append(robot_state.dof_forces.cpu())
                
                # rigid_body_contact_forces: [num_envs, num_bodies, 3]
                if robot_state.rigid_body_contact_forces is not None:
                    collected_data["rigid_body_contact_forces"].append(robot_state.rigid_body_contact_forces.cpu())

                if hasattr(robot_state, "rigid_body_pos") and robot_state.rigid_body_pos is not None:
                    collected_data["rigid_body_pos"].append(robot_state.rigid_body_pos.cpu())
                
                collected_data["root_pos"].append(robot_state.root_pos.cpu())
                collected_data["root_rot"].append(robot_state.root_rot.cpu())
                collected_data["root_vel"].append(robot_state.root_vel.cpu())
                collected_data["root_ang_vel"].append(robot_state.root_ang_vel.cpu())

                # Get IMU data
                if hasattr(env.simulator, "get_imu_data"):
                    imu_data = env.simulator.get_imu_data()
                    if imu_data is not None:
                        collected_data["imu_orientation"].append(imu_data["orientation"].cpu())
                        collected_data["imu_ang_vel"].append(imu_data["ang_vel"].cpu())
                        collected_data["imu_lin_acc"].append(imu_data["lin_acc"].cpu())

                # 3. Agent Step
                model_outs = agent.model(obs_td)
                if "mean_action" in model_outs:
                    actions = model_outs["mean_action"]
                else:
                    actions = model_outs["action"]

                obs, rewards, dones, terminated, extras = env.step(actions)
                obs = agent.add_agent_info_to_obs(obs)
                obs_td = agent.obs_dict_to_tensordict(obs)
                
                collected_data["rewards"].append(rewards.cpu())
                collected_data["dones"].append(dones.cpu())

        # Consolidate data
        log.info("Consolidating collected data...")
        for key in collected_data:
            if len(collected_data[key]) > 0:
                collected_data[key] = torch.stack(collected_data[key])
            else:
                collected_data[key] = None

        # Add metadata
        collected_data["fps"] = env.simulator.config.sim.fps
        collected_data["dt"] = env.simulator.dt
        collected_data["body_names"] = env.simulator._body_names
        collected_data["dof_names"] = env.simulator._dof_names

        # Derive per-foot GRF/COP if we have pad contact forces + positions
        try:
            body_names = list(collected_data["body_names"])

            contact_forces = collected_data.get("rigid_body_contact_forces")
            body_pos = collected_data.get("rigid_body_pos")

            if isinstance(contact_forces, torch.Tensor) and isinstance(body_pos, torch.Tensor):
                name_to_idx = {name: i for i, name in enumerate(body_names)}
                left_pad_names = ["L_Heel", "L_MetMedial", "L_MetLateral", "L_ToeTip"]
                right_pad_names = ["R_Heel", "R_MetMedial", "R_MetLateral", "R_ToeTip"]

                def _indices_for(names: list[str]) -> list[int]:
                    return [name_to_idx[n] for n in names if n in name_to_idx]

                left_ids = _indices_for(left_pad_names)
                right_ids = _indices_for(right_pad_names)

                eps = 1e-8

                def _foot_grf_cop(ids: list[int]):
                    # contact_forces: [T, E, B, 3]; body_pos: [T, E, B, 3]
                    f = contact_forces[:, :, ids, :]
                    p = body_pos[:, :, ids, :]

                    grf = f.sum(dim=2)  # [T, E, 3]

                    # Use vertical force as weight for COP.
                    # (Assumes +Z is up, which is consistent with the USD assets.)
                    fz = torch.relu(f[..., 2])  # [T, E, P]
                    fz_sum = fz.sum(dim=2)  # [T, E]

                    cop = (p * fz.unsqueeze(-1)).sum(dim=2) / (fz_sum.unsqueeze(-1).clamp_min(eps))
                    cop = torch.where(
                        (fz_sum > eps).unsqueeze(-1),
                        cop,
                        torch.full_like(cop, float("nan")),
                    )
                    return grf, cop, fz_sum

                if left_ids:
                    grf_l, cop_l, fz_l = _foot_grf_cop(left_ids)
                    collected_data["grf_left"] = grf_l
                    collected_data["cop_left"] = cop_l
                    collected_data["fz_left"] = fz_l

                if right_ids:
                    grf_r, cop_r, fz_r = _foot_grf_cop(right_ids)
                    collected_data["grf_right"] = grf_r
                    collected_data["cop_right"] = cop_r
                    collected_data["fz_right"] = fz_r
        except Exception as e:
            log.warning(f"Failed to compute COP/GRF from contact pads: {e}")

        # Save to file
        save_path = output_dir / "biomechanics_data.pt"
        torch.save(collected_data, save_path)
        log.info(f"Saved raw biomechanics data to {save_path}")

        # Optional: Save as .npz for easier loading without PyTorch/Project deps
        import numpy as np
        np_data = {}
        for key, value in collected_data.items():
            if isinstance(value, torch.Tensor):
                np_data[key] = value.numpy()
            elif isinstance(value, list):
                np_data[key] = np.array(value)
            else:
                np_data[key] = value
        
        np_save_path = output_dir / "biomechanics_data.npz"
        np.savez_compressed(np_save_path, **np_data)
        log.info(f"Saved numpy biomechanics data to {np_save_path}")


if __name__ == "__main__":
    main()
