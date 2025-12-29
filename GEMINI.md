# Gemini Project Analysis: ProtoMotions 3

## Project Overview

This repository contains **ProtoMotions 3**, a GPU-accelerated simulation and learning framework designed for training physically simulated digital humans and humanoid robots. It serves as a rapid prototyping platform for researchers and practitioners in animation, robotics, and reinforcement learning.

The framework is built on Python and leverages powerful simulation engines and deep learning libraries.

- **Core Technologies:**
    - **Programming Language:** Python (>=3.8)
    - **Deep Learning:** PyTorch, PyTorch Lightning
    - **Simulation Engines:** NVIDIA Isaac Lab, NVIDIA Newton (beta)
    - **Experiment Tracking:** Weights & Biases (`wandb`)
    - **Configuration:** A custom, robust configuration system inspired by Hydra. It saves fully resolved configurations as pickled objects to ensure reproducibility.

- **Key Features:**
    - Large-scale, multi-GPU training for motion skills using datasets like AMASS.
    - One-command motion retargeting from human data to robots using a PyRoki-based optimizer.
    - A modular architecture allowing for custom robots, environments, RL algorithms, and simulators.
    - Sim-to-sim policy transfer and high-fidelity rendering with IsaacSim.
    - Support for procedural scene generation and generative policies.

## NVIDIA Newton Physics Simulator

NVIDIA Newton is a GPU-accelerated physics simulator built on NVIDIA Warp, specifically leveraging MuJoCo Warp. It is currently in a beta phase of development.

-   **Integration**: Newton is fully integrated into ProtoMotions as a selectable simulator backend, alongside IsaacGym, IsaacLab, and Genesis. It can be activated using the `--simulator newton` command-line argument in various training and example scripts.
-   **Configuration**: Newton-specific simulation parameters are managed through `NewtonSimParams` and `NewtonSimulatorConfig` (located in `protomotions/simulator/newton/config.py`). Robot configurations can also include Newton-specific settings.
-   **Testing**: A dedicated test suite, `protomotions/tests/test_newton_simulator_fk.py`, is available to validate the Newton installation and its forward kinematics integration. Note that there might be minor differences in results compared to MotionLib.
-   **Installation**:
    1.  Clone the Newton repository: `git clone git@github.com:newton-physics/newton.git`
    2.  Navigate to the Newton directory: `cd newton`
    3.  Create and activate a virtual environment (e.g., using `uv`): `uv venv` and `. .venv/bin/activate` (or `. .venv\Scripts\activate` on Windows).
    4.  Install Newton dependencies and then ProtoMotions Newton requirements: `pip install -e .` followed by `uv pip install -r /path/to/protomotions/requirements_newton.txt`. Refer to the official [Newton Installation Guide](https://newton-physics.github.io/newton/guide/installation.html) for complete details.
-   **Current Status**: As Newton is still in beta, users may encounter physics artifacts. Contributions to improve its fidelity are welcomed.

## Building and Running

### Installation

**Environment Setup:**
This project utilizes the `uv` package manager and a virtual environment named `Biomotions`. Ensure you are working within this environment on your Windows machine.

```powershell
# Activate the Biomotions environment (Windows)
# Assuming standard uv venv location
. \Biomotions\Scripts\activate
```

The project is installed as a Python package. The core dependencies are listed in several `requirements_*.txt` files, corresponding to different simulation environments. For the Isaac Lab environment, the dependencies can be installed from `requirements_isaaclab.txt`.

```bash
# Example installation for the Isaac Lab environment
pip install -r requirements_isaaclab.txt
python setup.py install
```

**Note:** Specific simulators like Isaac Lab have their own installation procedures which must be completed first. Refer to the official documentation for those simulators.

### Running Experiments

The main entry point for training is `protomotions/train_agent.py`. This script requires several arguments to define the experiment.

**Training Command Structure:**

```bash
python protomotions/train_agent.py \\
    --robot-name <ROBOT_NAME> \\
    --simulator <SIMULATOR_NAME> \\
    --experiment-path <PATH_TO_EXPERIMENT_FILE> \\
    --experiment-name <YOUR_EXPERIMENT_NAME> \\
    --motion-file <PATH_TO_MOTION_DATA> \\
    --num-envs <NUMBER_OF_ENVIRONMENTS> \\
    --batch-size <TRAINING_BATCH_SIZE> \\
    [--use-wandb] \\
    [--checkpoint /path/to/checkpoint.ckpt]
```

**Example Training Run:**

```bash
# This is a conceptual example. You will need to provide a valid motion file.
python protomotions/train_agent.py \\
    --robot-name g1 \\
    --simulator isaaclab \\
    --experiment-path examples/experiments/mimic/mlp.py \\
    --experiment-name g1_mimic_example \\
    --motion-file /path/to/your/motions.pt \\
    --num-envs 4096 \\
    --batch-size 16384
```

### Running Examples and Tutorials

The `examples/` directory contains various tutorials and standalone scripts. These are typically run directly and accept command-line arguments to specify the robot and simulator.

**Example Tutorial Run:**

```bash
python examples/tutorial/4_basic_environment.py --simulator isaaclab --robot g1
```

## Development Conventions

- **Code Style:** The codebase does not appear to have a strict, enforced linter, but follows standard PEP 8 conventions.
- **Configuration:** Experiments are defined via Python files (e.g., in `examples/experiments/`) which construct configuration objects. The main training script (`protomotions/train_agent.py`) consumes these and allows overrides via the command line. On the first run, all configurations are saved to the `results/<experiment_name>` directory for reproducibility.
- **Reproducibility:** To resume an experiment exactly, the framework loads the saved `resolved_configs.pt` from the experiment's results directory. This ensures that the exact same configuration is used, and any new command-line overrides are ignored. For a new run with modified configs but starting from an old checkpoint, use the `--checkpoint` flag.
- **Contributions:** All contributions require a "sign-off" on commits, as detailed in `CONTRIBUTING.md`, to certify the Developer Certificate of Origin (DCO).
- **Testing:** There is a `protomotions/tests` directory, but no explicit instructions were found for running tests.
