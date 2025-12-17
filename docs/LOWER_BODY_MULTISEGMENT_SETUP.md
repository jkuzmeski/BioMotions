# Lower Body Multi-Segment Robot Setup Guide

This guide explains how to use the `lower_body_multisegment` robot with ProtoMotions.

## Robot Overview

The lower body multi-segment model features detailed foot biomechanics with the following structure:

### Kinematic Chain
```
Pelvis (freejoint - 6 DOF for position/orientation)
├── L_Hip (3 DOF: x, y, z)
│   └── L_Knee (3 DOF: x, y, z)
│       └── L_Ankle (3 DOF: x, y, z)
│           ├── L_Heel (3 DOF: x, y, z)
│           ├── L_MedToes (3 DOF: x, y, z)
│           └── L_LatMeta (3 DOF: x, y, z)
│               └── L_LatToes (3 DOF: x, y, z)
└── R_Hip (3 DOF: x, y, z)
    └── R_Knee (3 DOF: x, y, z)
        └── R_Ankle (3 DOF: x, y, z)
            ├── R_Heel (3 DOF: x, y, z)
            ├── R_MedToes (3 DOF: x, y, z)
            └── R_LatMeta (3 DOF: x, y, z)
                └── R_LatToes (3 DOF: x, y, z)
```

### Total Degrees of Freedom
- **48 actuated DOF** (3 per joint × 16 joints)
- Freejoint on pelvis (6 DOF for base pose, not actuated)

### Foot Segments
Each foot has 4 independent contact segments:
- **Heel**: Posterior foot contact during heel strike
- **MedToes**: Medial forefoot and toes (hallux side)
- **LatMeta**: Lateral metatarsals (midfoot)
- **LatToes**: Lateral toes

## Files Created

| File | Description |
|------|-------------|
| `protomotions/robot_configs/lower_body_multisegment.py` | Robot configuration class |
| `protomotions/data/assets/mjcf/lower_body_multisegment.xml` | MuJoCo physics model (already provided) |
| `examples/experiments/mimic/lower_body_multisegment.py` | Example training configuration |

## Usage

### 1. Verify Setup

Test that the robot configuration loads correctly:

```python
from protomotions.robot_configs.factory import robot_config

# Load the robot configuration
cfg = robot_config("lower_body_multisegment")

# Check basic properties
print(f"Number of DOF: {cfg.kinematic_info.num_dofs}")
print(f"Number of bodies: {len(cfg.kinematic_info.body_names)}")
print(f"Body names: {cfg.kinematic_info.body_names}")
print(f"Joint names: {cfg.kinematic_info.joint_names}")
```

### 2. Convert MJCF to USD (for IsaacLab)

If you want to use IsaacLab/IsaacSim, you need to generate a USD file:

```bash
# From the project root with IsaacLab installed
python data/scripts/convert_mjcf_to_usd.py \
    protomotions/data/assets/mjcf/lower_body_multisegment.xml \
    protomotions/data/assets/usd/lower_body_multisegment.usda
```

### 3. Prepare Motion Data

You need motion data retargeted to this robot's skeleton. The motion file should be a `.pt` file containing:
- `rb_pos`: Rigid body positions [num_motions, num_frames, num_bodies, 3]
- `rb_rot`: Rigid body rotations (quaternions) [num_motions, num_frames, num_bodies, 4]
- `dof_pos`: Joint positions [num_motions, num_frames, 48]
- `fps`: Frames per second

### 4. Kinematic Playback (Visualization)

Visualize reference motions without training:

```bash
python examples/env_kinematic_playback.py \
    --robot-name=lower_body_multisegment \
    --simulator=isaacgym \
    --num-envs=16 \
    --motion-file=path/to/your/motion.pt \
    --experiment-path=examples/experiments/mimic/lower_body_multisegment.py
```

### 5. Training

Train a mimic policy:

```bash
python protomotions/train_agent.py \
    --robot-name=lower_body_multisegment \
    --simulator=isaacgym \
    --num-envs=4096 \
    --motion-file=path/to/your/motion.pt \
    --experiment-path=examples/experiments/mimic/lower_body_multisegment.py \
    --experiment-name=multisegment_mimic_v1
```

### 6. Inference

Run a trained policy:

```bash
python protomotions/inference_agent.py \
    --robot-name=lower_body_multisegment \
    --simulator=isaacgym \
    --num-envs=16 \
    --motion-file=path/to/your/motion.pt \
    --experiment-path=examples/experiments/mimic/lower_body_multisegment.py \
    --checkpoint=results/multisegment_mimic_v1/checkpoints/best.pt
```

## Configuration Reference

### Control Parameters

The robot uses torque control by default with joint-specific PD gains:

| Joint Group | Stiffness | Damping | Effort Limit |
|-------------|-----------|---------|--------------|
| Hip | 800 | 80 | 500 |
| Knee | 800 | 80 | 500 |
| Ankle | 800 | 80 | 500 |
| Heel | 500 | 50 | 200 |
| MedToes | 500 | 50 | 100 |
| LatMeta | 500 | 50 | 100 |
| LatToes | 500 | 50 | 50 |

### Simulation Parameters

| Simulator | FPS | Decimation | Notes |
|-----------|-----|------------|-------|
| IsaacGym | 60 | 2 | Default for training |
| IsaacLab | 200 | 1 | Higher fidelity |
| Genesis | 60 | 2 | Community simulator |
| Newton | 120 | 4 | MuJoCo Warp based |

## Customization

### Changing Control Type

Edit the control type in `lower_body_multisegment.py`:

```python
control: ControlConfig = field(
    default_factory=lambda: ControlConfig(
        control_type=ControlType.BUILT_IN_PD,  # or TORQUE, PROPORTIONAL
        ...
    )
)
```

### Adding Contact Bodies

Add more bodies to contact sensing for GRF analysis:

```python
contact_bodies: List[str] = field(
    default_factory=lambda: [
        "L_Heel", "L_MedToes", "L_LatMeta", "L_LatToes",
        "R_Heel", "R_MedToes", "R_LatMeta", "R_LatToes",
        # Add more bodies here if needed
    ]
)
```

### Modifying PD Gains

Override control info for specific joints:

```python
override_control_info={
    "L_Hip_.*": ControlInfo(
        stiffness=1000.0,  # Increase stiffness
        damping=100.0,
        effort_limit=600,
        velocity_limit=100,
        armature=0.02,
    ),
    ...
}
```

## Troubleshooting

### Common Issues

1. **"Invalid robot name"**: Make sure `factory.py` is updated with the robot registration.

2. **USD file not found**: Run the MJCF to USD conversion script (requires IsaacLab).

3. **Motion dimension mismatch**: Ensure your motion data has 48 DOF (matching this robot's joint count).

4. **Simulation instability**: Try reducing the simulation step size or increasing substeps.

### Debug Mode

Enable detailed logging:

```bash
python protomotions/train_agent.py \
    --robot-name=lower_body_multisegment \
    ... \
    --debug
```

## Contact Sensors for Biomechanics

This robot is designed for detailed biomechanical analysis with contact sensors on each foot segment. You can extract:

- **Center of Pressure (COP)**: From contact forces on foot segments
- **Ground Reaction Forces (GRF)**: Total force on each foot
- **Foot roll-off pattern**: Heel strike → midfoot → toe-off

See `scripts/analyze_biomechanics.py` for examples of biomechanical analysis.
