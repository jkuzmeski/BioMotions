# Gemini Project Analysis: ProtoMotions 3

## Project Overview

This repository contains **ProtoMotions 3**, a GPU-accelerated simulation and learning framework designed for training physically simulated digital humans and humanoid robots. It serves as a rapid prototyping platform for researchers and practitioners in animation, robotics, and reinforcement learning.

The framework is built on Python and leverages powerful simulation engines and deep learning libraries.

- **Core Technologies:**
    - **Programming Language:** Python (>=3.8)
    - **Deep Learning:** PyTorch, PyTorch Lightning
    - **Simulation Engines:** NVIDIA IsaacGym, NVIDIA Isaac Lab, NVIDIA Newton, Genesis
    - **Experiment Tracking:** Weights & Biases (`wandb`)
    - **Configuration:** A custom, robust configuration system inspired by Hydra. It saves fully resolved configurations as pickled objects to ensure reproducibility.

- **Key Features:**
    - Large-scale, multi-GPU training for motion skills using datasets like AMASS.
    - One-command motion retargeting from human data to robots using a PyRoki-based optimizer.
    - A modular architecture allowing for custom robots, environments, RL algorithms, and simulators.
    - Sim-to-sim policy transfer and high-fidelity rendering with IsaacSim.
    - Support for procedural scene generation and generative policies.

## Building and Running

### Installation

The project is installed as a Python package. The core dependencies are listed in several `requirements_*.txt` files, corresponding to different simulation environments. For the standard IsaacGym environment, the dependencies can be installed from `requirements_isaacgym.txt`.

```bash
# Example installation for the IsaacGym environment
pip install -r requirements_isaacgym.txt
python setup.py install
```

**Note:** Specific simulators like IsaacGym and Isaac Lab have their own installation procedures which must be completed first. Refer to the official documentation for those simulators.

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
    --simulator isaacgym \\
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
python examples/tutorial/4_basic_environment.py --simulator isaacgym --robot g1
```

## Development Conventions

- **Code Style:** The codebase does not appear to have a strict, enforced linter, but follows standard PEP 8 conventions.
- **Configuration:** Experiments are defined via Python files (e.g., in `examples/experiments/`) which construct configuration objects. The main training script (`protomotions/train_agent.py`) consumes these and allows overrides via the command line. On the first run, all configurations are saved to the `results/<experiment_name>` directory for reproducibility.
- **Reproducibility:** To resume an experiment exactly, the framework loads the saved `resolved_configs.pt` from the experiment's results directory. This ensures that the exact same configuration is used, and any new command-line overrides are ignored. For a new run with modified configs but starting from an old checkpoint, use the `--checkpoint` flag.
- **Contributions:** All contributions require a "sign-off" on commits, as detailed in `CONTRIBUTING.md`, to certify the Developer Certificate of Origin (DCO).
- **Testing:** There is a `protomotions/tests` directory, but no explicit instructions were found for running tests.
