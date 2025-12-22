"""Inspect the structure of a motion file."""
import torch
import sys

if len(sys.argv) > 1:
    motion_path = sys.argv[1]
else:
    motion_path = "d:/Isaac/BioMotions/data/motions/synthetic/straight_d5.0m_v1.5ms_h0deg.motion"

m = torch.load(motion_path, weights_only=False)

print("Keys:", list(m.keys()))
print("\nShapes:")
for k, v in m.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

# Check the rigid_body_pos structure - body 0 should be pelvis/root
print("\n--- Rigid Body Position Details ---")
print(f"rigid_body_pos shape: {m['rigid_body_pos'].shape}")
print(f"  Frame 0, Body 0 (root/pelvis): {m['rigid_body_pos'][0, 0]}")
print(f"  Frame 0, all bodies X: {m['rigid_body_pos'][0, :, 0]}")
print(f"  Frame 0, all bodies Y: {m['rigid_body_pos'][0, :, 1]}")
print(f"  Frame 0, all bodies Z: {m['rigid_body_pos'][0, :, 2]}")

print("\n--- Rigid Body Rotation Details ---")
print(f"rigid_body_rot shape: {m['rigid_body_rot'].shape}")
print(f"  Frame 0, Body 0 (root/pelvis): {m['rigid_body_rot'][0, 0]}")

print("\n--- DOF Details ---")
if 'dof_pos' in m:
    print(f"dof_pos shape: {m['dof_pos'].shape}")
if 'dof_vel' in m:
    print(f"dof_vel shape: {m['dof_vel'].shape}")

print("\n--- FPS ---")
if 'fps' in m:
    print(f"fps: {m['fps']}")
