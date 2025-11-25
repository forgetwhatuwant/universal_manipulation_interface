#!/usr/bin/env python3
"""
Test script to verify Piper coordinate frame transformation.

This script:
1. Tests movements in each UMI axis direction
2. Verifies the transformation is correct
3. Allows interactive testing and adjustment
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import time
import click
from scipy.spatial.transform import Rotation as R
from umi.real_world.piper_interface import PiperInterface
from umi.common.pose_util import pose_to_mat, mat_to_pose


def test_axis_movement(piper, initial_pose, axis_idx, axis_name, expected_direction, distance=0.02):
    """
    Test movement in a specific axis and verify direction.
    
    Args:
        piper: PiperInterface instance
        initial_pose: Starting pose
        axis_idx: 0=X, 1=Y, 2=Z
        axis_name: Name of axis for display
        expected_direction: Description of expected movement
        distance: Movement distance in meters
    """
    print(f"\n{'='*80}")
    print(f"Testing {axis_name}-axis movement (+{distance*100:.1f}cm)")
    print(f"Expected: {expected_direction}")
    print(f"{'='*80}")
    
    # Get initial position
    pos_before = initial_pose[:3].copy()
    print(f"Position before: [{pos_before[0]:.4f}, {pos_before[1]:.4f}, {pos_before[2]:.4f}]")
    
    # Create test pose
    test_pose = initial_pose.copy()
    test_pose[axis_idx] += distance
    
    # Send command
    print(f"Sending command: {axis_name} = {test_pose[axis_idx]:.4f}")
    piper.update_desired_ee_pose(test_pose)
    
    # Wait for movement
    print("Waiting 3 seconds for movement...")
    time.sleep(3.0)
    
    # Read actual position
    actual_pose = piper.get_ee_pose()
    pos_after = actual_pose[:3]
    print(f"Position after:  [{pos_after[0]:.4f}, {pos_after[1]:.4f}, {pos_after[2]:.4f}]")
    
    # Calculate movement
    movement = pos_after - pos_before
    movement_magnitude = np.linalg.norm(movement)
    print(f"Actual movement: [{movement[0]:.4f}, {movement[1]:.4f}, {movement[2]:.4f}]")
    print(f"Movement magnitude: {movement_magnitude*100:.2f} cm")
    
    # Check if movement is in expected direction
    expected_movement = np.zeros(3)
    expected_movement[axis_idx] = distance
    
    # Calculate alignment
    if movement_magnitude > 1e-4:
        movement_normalized = movement / movement_magnitude
        expected_normalized = expected_movement / distance
        alignment = np.dot(movement_normalized, expected_normalized)
        print(f"Direction alignment: {alignment:.3f} (1.0 = perfect, -1.0 = opposite)")
        
        if alignment > 0.9:
            print("✓ Movement is in correct direction!")
        elif alignment < -0.9:
            print("✗ Movement is in OPPOSITE direction!")
        else:
            print("⚠ Movement is in unexpected direction (not aligned with axis)")
    else:
        print("⚠ No significant movement detected")
    
    # Return to initial pose
    print(f"\nReturning to initial pose...")
    piper.update_desired_ee_pose(initial_pose)
    time.sleep(2.0)
    
    return movement, alignment if movement_magnitude > 1e-4 else 0.0


@click.command()
@click.option('--can_name', default='can0', help='CAN interface name (e.g., can0)')
@click.option('--distance', default=0.02, type=float, help='Movement distance in meters (default: 0.02 = 2cm)')
@click.option('--test_all', is_flag=True, default=False, help='Test all axes automatically')
def main(can_name, distance, test_all):
    """Test Piper coordinate frame transformation with actual movements."""
    
    print("=" * 80)
    print("Piper Coordinate Frame Transformation Test")
    print("=" * 80)
    
    # Initialize Piper interface
    print(f"\nConnecting to Piper robot on {can_name}...")
    piper = PiperInterface(can_name=can_name)
    
    try:
        # Get initial pose
        print("\nGetting initial EE pose...")
        initial_pose = piper.get_ee_pose()
        
        print(f"\nInitial EE Pose (UMI convention):")
        print(f"  Position (m): [{initial_pose[0]:.6f}, {initial_pose[1]:.6f}, {initial_pose[2]:.6f}]")
        print(f"  Rotation vector (rad): [{initial_pose[3]:.6f}, {initial_pose[4]:.6f}, {initial_pose[5]:.6f}]")
        
        # Get rotation matrix
        rot_vec = initial_pose[3:]
        rot = R.from_rotvec(rot_vec)
        rot_mat = rot.as_matrix()
        
        print(f"\nRotation Matrix (after transformation):")
        print(f"  X-axis: [{rot_mat[0,0]:.3f}, {rot_mat[1,0]:.3f}, {rot_mat[2,0]:.3f}]")
        print(f"  Y-axis: [{rot_mat[0,1]:.3f}, {rot_mat[1,1]:.3f}, {rot_mat[2,1]:.3f}]")
        print(f"  Z-axis: [{rot_mat[0,2]:.3f}, {rot_mat[1,2]:.3f}, {rot_mat[2,2]:.3f}]")
        
        # Check alignment
        base_x = np.array([1, 0, 0])
        base_y = np.array([0, 1, 0])
        base_z = np.array([0, 0, 1])
        
        x_axis = rot_mat[:, 0]
        y_axis = rot_mat[:, 1]
        z_axis = rot_mat[:, 2]
        
        print(f"\nAxis Alignment with Base Frame:")
        print(f"  X-axis with Base X (right):   {np.dot(x_axis, base_x):.3f}")
        print(f"  X-axis with Base Y (up):      {np.dot(x_axis, base_y):.3f}")
        print(f"  X-axis with Base Z (forward): {np.dot(x_axis, base_z):.3f}")
        print(f"  Y-axis with Base X (right):   {np.dot(y_axis, base_x):.3f}")
        print(f"  Y-axis with Base Y (up):      {np.dot(y_axis, base_y):.3f}")
        print(f"  Y-axis with Base Z (forward): {np.dot(y_axis, base_z):.3f}")
        print(f"  Z-axis with Base X (right):   {np.dot(z_axis, base_x):.3f}")
        print(f"  Z-axis with Base Y (up):      {np.dot(z_axis, base_y):.3f}")
        print(f"  Z-axis with Base Z (forward): {np.dot(z_axis, base_z):.3f}")
        
        if not test_all:
            print("\n" + "=" * 80)
            print("WARNING: Robot will move!")
            print("This will test movements in X, Y, Z axes.")
            print("Make sure the robot has clear space to move.")
            print("Press Enter to continue or Ctrl+C to cancel.")
            print("=" * 80)
            input()
        
        # Test each axis
        results = {}
        
        # Test X-axis (should move right)
        movement_x, align_x = test_axis_movement(
            piper, initial_pose, 0, "X", "move RIGHT (from robot's perspective)", distance
        )
        results['X'] = {'movement': movement_x, 'alignment': align_x}
        
        if not test_all:
            input("Press Enter to continue to Y-axis test...")
        
        # Test Y-axis (should move up)
        movement_y, align_y = test_axis_movement(
            piper, initial_pose, 1, "Y", "move UP", distance
        )
        results['Y'] = {'movement': movement_y, 'alignment': align_y}
        
        if not test_all:
            input("Press Enter to continue to Z-axis test...")
        
        # Test Z-axis (should move forward)
        movement_z, align_z = test_axis_movement(
            piper, initial_pose, 2, "Z", "move FORWARD (away from base)", distance
        )
        results['Z'] = {'movement': movement_z, 'alignment': align_z}
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"X-axis alignment: {align_x:.3f} (should be ≈ 1.0 for RIGHT)")
        print(f"Y-axis alignment: {align_y:.3f} (should be ≈ 1.0 for UP)")
        print(f"Z-axis alignment: {align_z:.3f} (should be ≈ 1.0 for FORWARD)")
        
        if align_x > 0.9 and align_y > 0.9 and align_z > 0.9:
            print("\n✓ All axes are correctly aligned!")
        else:
            print("\n✗ Some axes are NOT correctly aligned!")
            print("The transformation matrix may need adjustment.")
            
            if align_x < -0.9:
                print("  → X-axis is pointing LEFT instead of RIGHT")
            if align_z < -0.9:
                print("  → Z-axis is pointing BACKWARD instead of FORWARD")
        
        print("\n" + "=" * 80)
        
    finally:
        piper.close()
        print("\nDisconnected from robot.")


if __name__ == '__main__':
    main()

