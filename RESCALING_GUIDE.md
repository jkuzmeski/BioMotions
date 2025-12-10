# Height-Based Model Rescaling and Registration Guide

## Overview

The BioMotions project now includes an automated system for rescaling the SMPL lower body model to different subject heights and automatically registering them with ProtoMotions for use in training pipelines.

## What's Included

### 1. Quick Rescaling Script
**Location:** `biomechanics_retarget/quick_rescale.py`

Rapidly rescale models to new heights with automatic file placement in the proper asset folders.

```bash
# Rescale to a new height
python biomechanics_retarget/quick_rescale.py --height 156

# List available heights
python biomechanics_retarget/quick_rescale.py --list

# Preview without making changes
python biomechanics_retarget/quick_rescale.py --height 165 --dry-run
```

### 2. Robot Configuration System
**Location:** `protomotions/robot_configs/smpl_lower_body.py`

Provides automatic height-based robot configuration with:
- Height-parameterized factory (`SmplLowerBodyConfigFactory`)
- Pre-defined configurations for 156cm, 170cm, and 180cm
- Automatically scaled PD control gains
- Correct asset file paths pointing to `data/assets/`

### 3. Factory Registration
**Location:** `protomotions/robot_configs/factory.py`

Updated to support SMPL lower body variants:
- `smpl_lower_body` - Base 170cm model
- `smpl_lower_body_XXXcm` - Height-scaled models (e.g., `smpl_lower_body_156cm`)
- `smpl_lower_body_XXXcm_torque` - Torque control variants

### 4. Updated Pipeline
**Location:** `biomechanics_retarget/pipeline.py`

Enhanced to automatically select/create models based on subject height:

```bash
# Process with automatic height-based model
python biomechanics_retarget/pipeline.py \
    ./treadmill_data/S02 \
    ./processed_data/S02 \
    --height 156
```

## Asset File Organization

Rescaled models are automatically placed in the ProtoMotions asset directories:

```
protomotions/data/assets/
├── mjcf/
│   ├── smpl_humanoid_lower_body_adjusted_pd.xml (base)
│   ├── smpl_humanoid_lower_body_adjusted_pd_height_156cm.xml
│   ├── smpl_humanoid_lower_body_adjusted_pd_height_170cm.xml
│   └── smpl_humanoid_lower_body_adjusted_pd_height_180cm.xml
└── usd/
    ├── smpl_humanoid_lower_body_adjusted_pd.usda (base)
    ├── smpl_humanoid_lower_body_adjusted_pd_height_156cm.usda
    ├── smpl_humanoid_lower_body_adjusted_pd_height_170cm.usda
    └── smpl_humanoid_lower_body_adjusted_pd_height_180cm.usda
```

## Usage Examples

### Example 1: Quick Rescale for New Subject Height

```bash
# Move to biomechanics_retarget directory
cd biomechanics_retarget

# Rescale to 160cm
python quick_rescale.py --height 160

# Output shows:
# ✅ Created: smpl_humanoid_lower_body_adjusted_pd_height_160cm.xml
# ✅ Created: smpl_humanoid_lower_body_adjusted_pd_height_160cm.usda
# Robot name: smpl_lower_body_160cm
# Root height: 0.897m
```

### Example 2: Process Motions with Height-Based Model

```bash
# Process treadmill data for 156cm subject
python pipeline.py \
    ./treadmill_data/S02 \
    ./processed_data/S02 \
    --height 156 \
    --fps 200
```

The pipeline will:
1. Check if the 156cm model exists
2. Create it automatically if needed
3. Use it for the entire processing pipeline
4. Output ready-to-use MotionLib with correct scaling

### Example 3: Use in Training Code

```python
from protomotions.robot_configs.factory import robot_config

# All of these work automatically
config_156 = robot_config("smpl_lower_body_156cm")
config_180 = robot_config("smpl_lower_body_180cm")
config_base = robot_config("smpl_lower_body")  # 170cm

# Access robot properties
print(f"Root height: {config_156.default_root_height}m")  # 0.872m
print(f"Asset file: {config_156.asset.asset_file_name}")  # mjcf/...height_156cm.xml
```

### Example 4: Complete Workflow for New Subject

```bash
# Step 1: Rescale (one-time per height)
cd biomechanics_retarget
python quick_rescale.py --height 165

# Step 2: Process motion data
python pipeline.py \
    ./treadmill_data/subject_165cm \
    ./processed_data/subject_165cm \
    --height 165

# Step 3: Use in ProtoMotions training
cd ..
python protomotions/train_agent.py \
    robot=smpl_lower_body_165cm \
    sensor=default \
    # ... other training parameters
```

## Key Features

### Automatic Model Creation
When you specify a height that doesn't have a model yet, the pipeline automatically:
1. Rescales the base model to the target height
2. Places XML in `protomotions/data/assets/mjcf/`
3. Places USDA in `protomotions/data/assets/usd/`
4. Registers the configuration with the factory

### Height-Scaled Properties

The system automatically adjusts:
- **Body dimensions**: Proportional scaling of all body parts
- **Joint capsule sizes**: Scaled based on height
- **Root height**: Adjusted for correct standing position
- **PD control gains**: Set from base configuration

### Dry-Run Mode

Preview changes without modifying files:

```bash
python quick_rescale.py --height 175 --dry-run
```

Output example:
```
Quick Rescale: SMPL Lower Body to 175cm
Variant: adjusted_pd
Scale factor: 1.0294

📐 Rescaling XML to 175cm...
   [DRY RUN] Would create XML file

🔧 Generating USDA from scaled XML...
   [DRY RUN] Would create USDA file

Robot name: smpl_lower_body_175cm
Root height: 0.977m
```

## Available Heights

Pre-generated heights:
- **156cm** (shorter subjects)
- **170cm** (base/average)
- **180cm** (taller subjects)

Generate any height between 100-250cm on demand:

```bash
# Works for any height
python quick_rescale.py --height 152
python quick_rescale.py --height 168
python quick_rescale.py --height 195
```

## Technical Details

### Asset Path Resolution

Robot configs use `protomotions/data/assets` as the asset root. The factory creates correct paths:

```
asset_root: "protomotions/data/assets"
asset_file_name: "mjcf/smpl_humanoid_lower_body_adjusted_pd_height_156cm.xml"
```

Resolves to:
```
${PROJECT_ROOT}/protomotions/data/assets/mjcf/smpl_humanoid_lower_body_adjusted_pd_height_156cm.xml
```

### Scaling Mathematics

Height scaling is proportional:
- Base model height: 1.70m
- Scale factor: `target_height_m / 1.70`
- All Z-axis body positions and joint capsules scale equally

Root height calculation:
```
root_height = 0.95m * (target_height / 170cm)
```

### Model Variants

Currently supports:
- `adjusted_pd` - PD control tuned for stability (default)
- `adjusted_torque` - Torque limits adjusted for scaling

```bash
python quick_rescale.py --height 175 --variant adjusted_torque
```

## Troubleshooting

### Model Not Found Error

```
Error: No model for 165cm found
```

Solution: Generate it first

```bash
python quick_rescale.py --height 165
```

### Asset Path Issues

If you get path errors when loading the robot:

1. Check asset folder exists: `protomotions/data/assets/mjcf/`
2. Verify file is there: `ls protomotions/data/assets/mjcf/smpl_*_height_XXXcm.xml`
3. Ensure correct asset_root in config: `protomotions/data/assets`

### Configuration Not Registered

If factory doesn't recognize a new height:

1. Regenerate the model: `python quick_rescale.py --height XXX --force`
2. Verify files are in correct locations
3. Restart your Python environment to reload modules

## File Reference

### Core Scripts

| File | Purpose |
|------|---------|
| `biomechanics_retarget/quick_rescale.py` | Rescale models and manage heights |
| `biomechanics_retarget/pipeline.py` | Main processing pipeline with height support |
| `protomotions/robot_configs/smpl_lower_body.py` | Robot configuration definitions |
| `protomotions/robot_configs/factory.py` | Configuration factory with height parsing |

### Generated Assets

| Directory | Contents |
|-----------|----------|
| `protomotions/data/assets/mjcf/` | MJCF model files for each height |
| `protomotions/data/assets/usd/` | USD asset files for IsaacLab |
| `rescale/` | Base model files (source for rescaling) |

## Integration with Training

The height-scaled models integrate seamlessly with ProtoMotions training:

```bash
# Single subject
python train_agent.py robot=smpl_lower_body_156cm ...

# Multi-subject ensemble (same model, different heights)
python train_agent.py robot=smpl_lower_body_156cm,smpl_lower_body_170cm,smpl_lower_body_180cm ...
```

## Next Steps

1. **Rescale for your subjects**: `python quick_rescale.py --height XXX`
2. **Process motion data**: `python pipeline.py ... --height XXX`
3. **Train models**: Use `robot=smpl_lower_body_XXXcm` in training configs
4. **Evaluate**: Compare model performance across different heights
