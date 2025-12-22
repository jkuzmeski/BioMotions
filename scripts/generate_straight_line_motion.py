"""
Generate Synthetic Motion Files with Straight Line Pelvis Trajectories.

This script creates motion files where only the pelvis X and Y positions are
populated with meaningful data (straight line movements). All other data
(rotations, velocities, other body positions) are set to neutral/zero values
since they will be masked during inference.

The generated motions are compatible with the masked mimic system configured
to only use pelvis translation X,Y (constraint_state=3).

Usage:
    python generate_straight_line_motion.py --distance 5.0 --velocity 1.5 --fps 30
    python generate_straight_line_motion.py --distance 10.0 --velocity 2.0 --heading 45
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class StraightLineMotionConfig:
    """Configuration for generating straight line motion."""
    
    # Movement parameters
    distance: float = 5.0  # Total distance to travel (meters)
    velocity: float = 1.5  # Movement velocity (m/s)
    heading: float = 0.0   # Movement direction in degrees (0 = +X, 90 = +Y)
    
    # Starting position
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.95  # Default pelvis height for 170cm model
    
    # Motion file parameters
    fps: int = 30  # Frames per second
    
    # Robot configuration (SMPL lower body)
    num_bodies: int = 9   # Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe
    num_dofs: int = 24    # 8 joints * 3 DOF each
    
    # Output
    output_dir: str = "data/motions/synthetic"
    output_name: Optional[str] = None


def generate_straight_line_motion(config: StraightLineMotionConfig) -> dict:
    """
    Generate a synthetic motion file with straight line pelvis trajectory.
    
    The motion contains:
    - Pelvis X,Y positions following a straight line at constant velocity
    - All other data (rotations, velocities, other bodies) set to neutral values
    
    Args:
        config: Configuration for the motion generation
        
    Returns:
        Dictionary containing the motion data in the standard format
    """
    # Calculate motion duration and number of frames
    duration = config.distance / config.velocity
    num_frames = int(np.ceil(duration * config.fps)) + 1
    dt = 1.0 / config.fps
    
    print(f"Generating straight line motion:")
    print(f"  Distance: {config.distance:.2f} m")
    print(f"  Velocity: {config.velocity:.2f} m/s")
    print(f"  Duration: {duration:.2f} s")
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {config.fps}")
    print(f"  Heading: {config.heading:.1f} degrees")
    
    # Convert heading to radians
    heading_rad = np.deg2rad(config.heading)
    
    # Calculate direction vector
    dir_x = np.cos(heading_rad)
    dir_y = np.sin(heading_rad)
    
    # Generate time array
    times = torch.arange(num_frames, dtype=torch.float32) * dt
    
    # Generate pelvis positions along straight line
    # Position = start + velocity * time * direction
    pelvis_x = config.start_x + config.velocity * times * dir_x
    pelvis_y = config.start_y + config.velocity * times * dir_y
    pelvis_z = torch.full((num_frames,), config.start_z, dtype=torch.float32)
    
    # Initialize all tensors
    # Rigid body positions: [num_frames, num_bodies, 3]
    rigid_body_pos = torch.zeros(num_frames, config.num_bodies, 3, dtype=torch.float32)
    
    # Set pelvis (body 0) positions
    rigid_body_pos[:, 0, 0] = pelvis_x
    rigid_body_pos[:, 0, 1] = pelvis_y
    rigid_body_pos[:, 0, 2] = pelvis_z
    
    # Set other bodies relative to pelvis (simple offset for now)
    # These will be masked anyway, but having reasonable values helps visualization
    body_offsets = get_default_body_offsets()
    for body_idx in range(1, config.num_bodies):
        rigid_body_pos[:, body_idx, 0] = pelvis_x + body_offsets[body_idx][0]
        rigid_body_pos[:, body_idx, 1] = pelvis_y + body_offsets[body_idx][1]
        rigid_body_pos[:, body_idx, 2] = pelvis_z + body_offsets[body_idx][2]
    
    # Rigid body rotations: [num_frames, num_bodies, 4] (quaternion xyzw)
    # Identity quaternion = [0, 0, 0, 1] in xyzw format
    rigid_body_rot = torch.zeros(num_frames, config.num_bodies, 4, dtype=torch.float32)
    rigid_body_rot[:, :, 3] = 1.0  # w component = 1 for identity
    
    # Apply heading rotation to pelvis
    half_angle = heading_rad / 2
    rigid_body_rot[:, 0, 2] = np.sin(half_angle)  # z component
    rigid_body_rot[:, 0, 3] = np.cos(half_angle)  # w component
    
    # Rigid body velocities: [num_frames, num_bodies, 3]
    rigid_body_vel = torch.zeros(num_frames, config.num_bodies, 3, dtype=torch.float32)
    # Set pelvis velocity (constant)
    rigid_body_vel[:, 0, 0] = config.velocity * dir_x
    rigid_body_vel[:, 0, 1] = config.velocity * dir_y
    
    # Rigid body angular velocities: [num_frames, num_bodies, 3]
    rigid_body_ang_vel = torch.zeros(num_frames, config.num_bodies, 3, dtype=torch.float32)
    
    # DOF positions: [num_frames, num_dofs]
    # Zero = standing pose (all joints at rest)
    dof_pos = torch.zeros(num_frames, config.num_dofs, dtype=torch.float32)
    
    # DOF velocities: [num_frames, num_dofs]
    dof_vel = torch.zeros(num_frames, config.num_dofs, dtype=torch.float32)
    
    # Contacts: [num_frames, num_bodies]
    # Assume feet are in contact (bodies 3, 4, 7, 8 for SMPL lower body)
    rigid_body_contacts = torch.zeros(num_frames, config.num_bodies, dtype=torch.bool)
    rigid_body_contacts[:, 3] = True  # L_Ankle
    rigid_body_contacts[:, 4] = True  # L_Toe
    rigid_body_contacts[:, 7] = True  # R_Ankle
    rigid_body_contacts[:, 8] = True  # R_Toe
    
    # Assemble motion dictionary
    from protomotions.simulator.base_simulator.simulator_state import StateConversion
    
    motion_data = {
        'state_conversion': StateConversion.COMMON,
        'fps': config.fps,
        'dof_pos': dof_pos,
        'dof_vel': dof_vel,
        'rigid_body_pos': rigid_body_pos,
        'rigid_body_rot': rigid_body_rot,
        'rigid_body_vel': rigid_body_vel,
        'rigid_body_ang_vel': rigid_body_ang_vel,
        'rigid_body_contacts': rigid_body_contacts,
    }
    
    return motion_data


def get_default_body_offsets():
    """
    Get default body offsets relative to pelvis for SMPL lower body.
    
    Body order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe
    
    Returns:
        List of (x, y, z) offsets for each body relative to pelvis
    """
    return [
        (0.0, 0.0, 0.0),      # 0: Pelvis (reference)
        (0.0, 0.1, -0.1),     # 1: L_Hip
        (0.0, 0.1, -0.5),     # 2: L_Knee
        (0.0, 0.1, -0.9),     # 3: L_Ankle
        (0.1, 0.1, -0.95),    # 4: L_Toe
        (0.0, -0.1, -0.1),    # 5: R_Hip
        (0.0, -0.1, -0.5),    # 6: R_Knee
        (0.0, -0.1, -0.9),    # 7: R_Ankle
        (0.1, -0.1, -0.95),   # 8: R_Toe
    ]


def save_motion(motion_data: dict, output_path: Path):
    """Save motion data to a .motion file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion_data, output_path)
    print(f"Saved motion to: {output_path}")


def generate_motion_filename(config: StraightLineMotionConfig) -> str:
    """Generate a descriptive filename for the motion."""
    if config.output_name:
        return config.output_name
    
    return f"straight_d{config.distance:.1f}m_v{config.velocity:.1f}ms_h{config.heading:.0f}deg.motion"


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic straight line motion files for masked mimic"
    )
    
    # Movement parameters
    parser.add_argument(
        "--distance", type=float, default=5.0,
        help="Total distance to travel in meters (default: 5.0)"
    )
    parser.add_argument(
        "--velocity", type=float, default=1.5,
        help="Movement velocity in m/s (default: 1.5)"
    )
    parser.add_argument(
        "--heading", type=float, default=0.0,
        help="Movement direction in degrees, 0=+X, 90=+Y (default: 0.0)"
    )
    
    # Starting position
    parser.add_argument(
        "--start-x", type=float, default=0.0,
        help="Starting X position (default: 0.0)"
    )
    parser.add_argument(
        "--start-y", type=float, default=0.0,
        help="Starting Y position (default: 0.0)"
    )
    parser.add_argument(
        "--start-z", type=float, default=0.95,
        help="Starting Z position / pelvis height (default: 0.95)"
    )
    
    # Motion parameters
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second (default: 30)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="data/motions/synthetic",
        help="Output directory (default: data/motions/synthetic)"
    )
    parser.add_argument(
        "--output-name", type=str, default=None,
        help="Output filename (auto-generated if not specified)"
    )
    
    # Batch generation
    parser.add_argument(
        "--batch", action="store_true",
        help="Generate multiple motions with varying parameters"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Generate multiple motions
        generate_batch_motions(args)
    else:
        # Generate single motion
        config = StraightLineMotionConfig(
            distance=args.distance,
            velocity=args.velocity,
            heading=args.heading,
            start_x=args.start_x,
            start_y=args.start_y,
            start_z=args.start_z,
            fps=args.fps,
            output_dir=args.output_dir,
            output_name=args.output_name,
        )
        
        motion_data = generate_straight_line_motion(config)
        
        filename = generate_motion_filename(config)
        output_path = Path(config.output_dir) / filename
        save_motion(motion_data, output_path)
        
        # Print summary
        print_motion_summary(motion_data)


def generate_batch_motions(args):
    """Generate a batch of motions with varying parameters."""
    
    # Define parameter ranges
    distances = [100, 200, 300, 400, 500]
    velocities = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    headings = [0.0]  # Start with straight forward only
    
    print(f"Generating batch of {len(distances) * len(velocities) * len(headings)} motions...")
    
    for distance in distances:
        for velocity in velocities:
            for heading in headings:
                config = StraightLineMotionConfig(
                    distance=distance,
                    velocity=velocity,
                    heading=heading,
                    start_z=args.start_z,
                    fps=args.fps,
                    output_dir=args.output_dir,
                )
                
                motion_data = generate_straight_line_motion(config)
                
                filename = generate_motion_filename(config)
                output_path = Path(config.output_dir) / filename
                save_motion(motion_data, output_path)
    
    print(f"\nBatch generation complete!")


def print_motion_summary(motion_data: dict):
    """Print a summary of the generated motion."""
    print("\n--- Motion Summary ---")
    for key, value in motion_data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # Print trajectory info
    pos = motion_data['rigid_body_pos']
    print(f"\n--- Pelvis Trajectory ---")
    print(f"  Start: ({pos[0, 0, 0]:.3f}, {pos[0, 0, 1]:.3f}, {pos[0, 0, 2]:.3f})")
    print(f"  End:   ({pos[-1, 0, 0]:.3f}, {pos[-1, 0, 1]:.3f}, {pos[-1, 0, 2]:.3f})")
    
    total_dist = torch.sqrt(
        (pos[-1, 0, 0] - pos[0, 0, 0])**2 + 
        (pos[-1, 0, 1] - pos[0, 0, 1])**2
    )
    print(f"  Total XY distance: {total_dist:.3f} m")


if __name__ == "__main__":
    main()
