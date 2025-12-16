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
"""Smoke-test foot contact pads under torque control.

This script is intentionally minimal:
- Spawns a BaseEnv with a flat terrain and an empty scene.
- Uses the contact-pad SMPL lower-body robot variant.
- Steps with torque actions and prints per-pad vertical forces (Fz), plus
    a simple GRF/COP estimate.

Example:
    python scripts/smoke_test_contact_pads_torque.py --simulator isaacgym \
        --headless

Notes:
- "Torque control" here refers to the controller mode (ControlType.TORQUE).
    The SMPL lower-body configs in this repo already use torque control by
    default.
- Contact pad forces require the contact-pad robot variant:
  robot=smpl_lower_body_170cm_contact_pads
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test contact pads with torque control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator backend: isaacgym | isaaclab | newton | genesis",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="smpl_lower_body_170cm_contact_pads",
        help="Robot name (must include _contact_pads for pad bodies)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Headless mode (recommended for IsaacLab)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Use CPU only (if supported by backend)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print forces every N sim steps",
    )
    parser.add_argument(
        "--fz-threshold",
        type=float,
        default=20.0,
        help="Threshold (N) for considering a pad 'in contact'",
    )
    parser.add_argument(
        "--action-std",
        type=float,
        default=0.25,
        help="Std-dev of random torque actions after warmup",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Import simulator before torch (IsaacGym/IsaacLab requirement)
    from protomotions.utils.simulator_imports import (
        import_simulator_before_torch,
    )

    AppLauncher = import_simulator_before_torch(args.simulator)

    import torch

    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.simulator.factory import simulator_config
    from protomotions.robot_configs.factory import robot_config
    from protomotions.utils.hydra_replacement import get_class

    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.terrains.terrain import Terrain

    from protomotions.components.scene_lib import (
        Scene,
        SceneLib,
        SceneLibConfig,
    )
    from protomotions.components.motion_lib import MotionLib

    device = torch.device("cpu") if args.cpu_only else torch.device("cuda:0")

    robot_cfg = robot_config(args.robot)
    print(f"Robot config: {type(robot_cfg).__name__}")
    print(f"Control type: {robot_cfg.control.control_type}")
    print(f"Contact bodies: {robot_cfg.contact_bodies}")

    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        extra_simulator_params["simulation_app"] = app_launcher.app

    sim_cfg = simulator_config(
        args.simulator,
        robot_cfg,
        headless=args.headless,
        num_envs=args.num_envs,
        experiment_name="smoke_test_contact_pads_torque",
    )

    env_cfg = EnvConfig(max_episode_length=max(1000, args.steps + 10))

    terrain_cfg = TerrainConfig()
    terrain = Terrain(
        config=terrain_cfg,
        num_envs=sim_cfg.num_envs,
        device=device,
    )

    scene_lib_cfg = SceneLibConfig(scene_file=None)
    scene_lib = SceneLib(
        config=scene_lib_cfg,
        num_envs=sim_cfg.num_envs,
        scenes=[Scene(objects=[])],
        device=device,
        terrain=terrain,
    )

    motion_lib = MotionLib.empty(device=device)

    SimulatorClass = get_class(sim_cfg._target_)
    simulator = SimulatorClass(
        config=sim_cfg,
        robot_config=robot_cfg,
        terrain=terrain,
        scene_lib=scene_lib,
        device=device,
        **extra_simulator_params,
    )

    env = BaseEnv(
        config=env_cfg,
        robot_config=robot_cfg,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    env_ids = torch.arange(env.num_envs, device=device)
    env.reset(env_ids)

    pad_names_left = ["L_Heel", "L_MetMedial", "L_MetLateral", "L_ToeTip"]
    pad_names_right = ["R_Heel", "R_MetMedial", "R_MetLateral", "R_ToeTip"]

    def _pad_indices(names: list[str]) -> list[int]:
        body_names = list(env.simulator._body_names)
        name_to_idx = {n: i for i, n in enumerate(body_names)}
        missing = [n for n in names if n not in name_to_idx]
        if missing:
            print(
                "Warning: missing pad bodies in simulator body list: "
                f"{missing}"
            )
        return [name_to_idx[n] for n in names if n in name_to_idx]

    left_ids = _pad_indices(pad_names_left)
    right_ids = _pad_indices(pad_names_right)

    if not left_ids and not right_ids:
        print("No pad bodies found. Did you pass a *_contact_pads robot?")
        print(f"robot={args.robot}")
        return 2

    eps = 1e-8

    for t in range(args.steps):
        if t < args.warmup:
            actions = torch.zeros(
                env.num_envs,
                robot_cfg.number_of_actions,
                device=device,
            )
        else:
            actions = torch.randn(
                env.num_envs, robot_cfg.number_of_actions, device=device
            ) * float(args.action_std)

        env.step(actions)

        if (t % args.print_every) != 0:
            continue

        state = env.simulator.get_robot_state()
        cf = getattr(state, "rigid_body_contact_forces", None)
        bp = getattr(state, "rigid_body_pos", None)

        if cf is None:
            print(
                "rigid_body_contact_forces is None "
                "(contact sensors not enabled?)"
            )
            return 3

        def _summarize(ids: list[int], label: str) -> None:
            if not ids:
                return
            # cf: [E, B, 3]
            f = cf[:, ids, :]
            fz = torch.relu(f[..., 2])  # [E, P]
            in_contact = (fz > float(args.fz_threshold)).float()
            fz_mean = fz.mean().item()
            fz_max = fz.max().item()
            contact_frac = in_contact.mean().item()
            msg = (
                f"t={t:05d} {label}: "
                f"fz_mean={fz_mean:8.2f}N "
                f"fz_max={fz_max:8.2f}N "
                f"contact%={100*contact_frac:5.1f}%"
            )
            print(msg)

        _summarize(left_ids, "LEFT")
        _summarize(right_ids, "RIGHT")

        # Simple COP/GRF estimate for env 0
        if bp is not None:
            env0 = 0

            def _grf_cop(ids: list[int]):
                if not ids:
                    return None
                f = cf[env0, ids, :]  # [P, 3]
                p = bp[env0, ids, :]  # [P, 3]
                grf = f.sum(dim=0)
                fz = torch.relu(f[:, 2])
                fz_sum = fz.sum().clamp_min(eps)
                cop = (p * fz.unsqueeze(-1)).sum(dim=0) / fz_sum
                return grf, cop, fz_sum

            out_l = _grf_cop(left_ids)
            out_r = _grf_cop(right_ids)
            if out_l is not None:
                grf, cop, fz_sum = out_l
                print(
                    "  env0 LEFT  "
                    f"GRF={grf.tolist()}  "
                    f"COP={cop.tolist()}  "
                    f"sum_fz={fz_sum.item():.2f}"
                )
            if out_r is not None:
                grf, cop, fz_sum = out_r
                print(
                    "  env0 RIGHT "
                    f"GRF={grf.tolist()}  "
                    f"COP={cop.tolist()}  "
                    f"sum_fz={fz_sum.item():.2f}"
                )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
