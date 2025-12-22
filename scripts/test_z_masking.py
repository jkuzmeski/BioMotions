"""
Sanity check for Z-masking logic in MaskedMimic.

This script verifies that the constraint_state=3 (Trans XY only)
correctly masks the Z coordinate while allowing X and Y through.
"""

import torch


def test_z_masking_logic():
    print("=" * 60)
    print("Running Sanity Check for Z-Masking Logic")
    print("=" * 60)
    
    # Setup dimensions
    num_envs = 2
    num_bodies = 3
    num_future_steps = 2
    device = 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  num_envs={num_envs}")
    print(f"  num_bodies={num_bodies}")
    print(f"  num_future_steps={num_future_steps}")

    # 1. Simulate Mask Generation
    print("\n" + "-" * 40)
    print("[1] Testing Mask Generation")
    print("-" * 40)
    
    # Active bodies: All active
    active_body_ids = torch.ones(num_envs, num_bodies, dtype=torch.bool, device=device)
    
    # Constraint States
    # Env 0: [0 (XYZ), 3 (XY only), 2 (Rot only)]
    # Env 1: [3 (XY only), 1 (Both), 0 (XYZ)]
    constraint_states = torch.tensor([
        [0, 3, 2],
        [3, 1, 0]
    ], device=device)
    
    print(f"\nConstraint States:\n{constraint_states}")
    print("  0 = Trans XYZ, 1 = Both, 2 = Rot only, 3 = Trans XY only")
    
    # --- FIXED Logic from _sample_new_body_masks ---
    trans_mask = (constraint_states <= 1) | (constraint_states == 3)
    translation_mask = trans_mask & active_body_ids
    rot_mask = (constraint_states >= 1) & (constraint_states != 3)
    rotation_mask = rot_mask & active_body_ids

    xyz_mask = torch.ones(
        num_envs,
        num_bodies,
        3,
        dtype=torch.bool,
        device=device,
    )
    # If state is 3 (Trans XY), mask Z (index 2)
    # Use proper broadcasting instead of advanced indexing
    z_mask_indices = (constraint_states == 3)
    xyz_mask[:, :, 2] = xyz_mask[:, :, 2] & ~z_mask_indices

    new_mask = torch.zeros(
        num_envs,
        num_bodies,
        2,
        3,
        dtype=torch.bool,
        device=device,
    )
    new_mask[:, :, 0, :] = translation_mask.unsqueeze(-1) & xyz_mask
    new_mask[:, :, 1, :] = rotation_mask.unsqueeze(-1)
    # ------------------------------------------------
    
    print(f"\nXYZ Mask (per body):\n{xyz_mask}")
    
    # Verify Env 0, Body 1 (State 3: Trans XY)
    e0_b1_trans = new_mask[0, 1, 0]
    e0_b1_rot = new_mask[0, 1, 1]
    
    print(f"\n--- Env 0, Body 1 (State 3 - Trans XY) ---")
    print(f"  Trans Mask: {e0_b1_trans.tolist()}")
    print(f"  Expected:   [True, True, False]")
    print(f"  Rot Mask:   {e0_b1_rot.tolist()}")
    print(f"  Expected:   [False, False, False]")
    
    assert e0_b1_trans.tolist() == [True, True, False], "State 3 Trans mask incorrect"
    assert e0_b1_rot.tolist() == [False, False, False], "State 3 Rot mask incorrect"
    print("  ✓ PASSED")

    # Verify Env 0, Body 0 (State 0: Trans XYZ)
    e0_b0_trans = new_mask[0, 0, 0]
    e0_b0_rot = new_mask[0, 0, 1]
    
    print(f"\n--- Env 0, Body 0 (State 0 - Trans XYZ) ---")
    print(f"  Trans Mask: {e0_b0_trans.tolist()}")
    print(f"  Expected:   [True, True, True]")
    print(f"  Rot Mask:   {e0_b0_rot.tolist()}")
    print(f"  Expected:   [False, False, False]")
    
    assert e0_b0_trans.tolist() == [True, True, True], "State 0 Trans mask incorrect"
    assert e0_b0_rot.tolist() == [False, False, False], "State 0 Rot mask incorrect"
    print("  ✓ PASSED")

    # Verify Env 1, Body 0 (State 3: Trans XY)
    e1_b0_trans = new_mask[1, 0, 0]
    e1_b0_rot = new_mask[1, 0, 1]
    
    print(f"\n--- Env 1, Body 0 (State 3 - Trans XY) ---")
    print(f"  Trans Mask: {e1_b0_trans.tolist()}")
    print(f"  Expected:   [True, True, False]")
    print(f"  Rot Mask:   {e1_b0_rot.tolist()}")
    print(f"  Expected:   [False, False, False]")
    
    assert e1_b0_trans.tolist() == [True, True, False], "Env1 State 3 Trans mask incorrect"
    assert e1_b0_rot.tolist() == [False, False, False], "Env1 State 3 Rot mask incorrect"
    print("  ✓ PASSED")

    # Verify Env 1, Body 1 (State 1: Both)
    e1_b1_trans = new_mask[1, 1, 0]
    e1_b1_rot = new_mask[1, 1, 1]
    
    print(f"\n--- Env 1, Body 1 (State 1 - Both) ---")
    print(f"  Trans Mask: {e1_b1_trans.tolist()}")
    print(f"  Expected:   [True, True, True]")
    print(f"  Rot Mask:   {e1_b1_rot.tolist()}")
    print(f"  Expected:   [True, True, True]")
    
    assert e1_b1_trans.tolist() == [True, True, True], "State 1 Trans mask incorrect"
    assert e1_b1_rot.tolist() == [True, True, True], "State 1 Rot mask incorrect"
    print("  ✓ PASSED")

    print("\n" + "=" * 60)
    print("✓ All mask generation tests PASSED!")
    print("=" * 60)

    # 2. Simulate Observation Application
    print("\n" + "-" * 40)
    print("[2] Testing Observation Application")
    print("-" * 40)
    
    # Create dummy observations filled with ones
    # Shape: [num_envs, num_future_steps, num_bodies, 2, 4, 3]
    # The 4 comes from: rel_pos(3) + abs_pos(3) + rel_rot(6) + abs_rot(6) = 12 / 3 = 4 sub-vectors
    obs = torch.ones(num_envs, num_future_steps, num_bodies, 2, 4, 3, device=device)
    
    # Expand mask to match obs
    # Mask shape: [num_envs, num_bodies, 2, 3] -> [num_envs, num_future_steps, num_bodies, 2, 1, 3]
    mask_expanded = new_mask.unsqueeze(1).unsqueeze(4).expand(
        num_envs, num_future_steps, num_bodies, 2, 1, 3
    )
    
    # Apply mask (broadcasting over the 4 sub-vectors)
    masked_obs = obs * mask_expanded
    
    # Check results for Env 0, Body 1 (State 3)
    e0_b1_masked_trans = masked_obs[0, 0, 1, 0]  # [4, 3]
    e0_b1_masked_rot = masked_obs[0, 0, 1, 1]    # [4, 3]
    
    print(f"\nEnv 0, Body 1 (State 3) - Masked Observation Values:")
    print(f"  Trans (4x3):\n{e0_b1_masked_trans}")
    print(f"  Rot (4x3):\n{e0_b1_masked_rot}")
    
    # Check Trans Z columns (index 2) are all 0
    z_cols = e0_b1_masked_trans[:, 2]
    print(f"\n  Trans Z columns: {z_cols.tolist()}")
    print(f"  Expected:        [0.0, 0.0, 0.0, 0.0]")
    assert torch.all(z_cols == 0), "Trans Z columns should be 0"
    print("  ✓ Z correctly masked")
    
    # Check Trans X, Y columns are all 1
    xy_cols = e0_b1_masked_trans[:, :2]
    assert torch.all(xy_cols == 1), "Trans X,Y columns should be 1"
    print("  ✓ X,Y correctly passed through")
    
    # Check Rot is all zeros
    assert torch.all(e0_b1_masked_rot == 0), "Rot should be all 0"
    print("  ✓ Rotation correctly masked")
    
    print("\n" + "=" * 60)
    print("✓ All observation application tests PASSED!")
    print("=" * 60)
    
    # 3. Test Final Concatenation Shape
    print("\n" + "-" * 40)
    print("[3] Testing Final Concatenation Shape")
    print("-" * 40)
    
    masked_obs_flat = masked_obs.view(num_envs, num_future_steps, num_bodies, 2, 12)
    
    # Mask for concat: [num_envs, num_future_steps, num_bodies, 2, 3]
    mask_for_concat = new_mask.unsqueeze(1).expand(num_envs, num_future_steps, num_bodies, 2, 3)
    
    # Concat: [..., 12] + [..., 3] = [..., 15]
    final_obs = torch.cat((masked_obs_flat, mask_for_concat), dim=-1)
    
    print(f"\n  Masked obs shape:      {masked_obs_flat.shape}")
    print(f"  Mask for concat shape: {mask_for_concat.shape}")
    print(f"  Final obs shape:       {final_obs.shape}")
    print(f"  Final obs last dim:    {final_obs.shape[-1]} (Expected: 15)")
    
    assert final_obs.shape[-1] == 15, "Final dimension incorrect"
    print("  ✓ Shape is correct")
    
    # Flatten for network input
    final_flat = final_obs.view(num_envs, num_future_steps, -1)
    per_step_size = num_bodies * 2 * 15  # bodies * (trans/rot) * (12 obs + 3 mask)
    print(f"\n  Flattened shape: {final_flat.shape}")
    print(f"  Per-step obs size: {per_step_size}")
    
    print("\n" + "=" * 60)
    print("✓ ALL SANITY CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_z_masking_logic()
