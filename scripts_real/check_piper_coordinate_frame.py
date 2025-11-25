#!/usr/bin/env python3
"""
Script to check Piper EE coordinate frame alignment.

This script:
1. Gets current EE pose from Piper SDK
2. Visualizes the coordinate frame axes
3. Compares with expected UMI convention (+Z forward, +Y up, +X right)
4. Tests movement in each axis to verify orientation
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import click
import time

from umi.real_world.piper_interface import PiperInterface


def draw_arrow_3d(ax, pos, direction, color='r', length=0.05, alpha=0.8):
    """Draw a 3D arrow representing orientation using quiver."""
    direction = direction / np.linalg.norm(direction) * length
    ax.quiver(pos[0], pos[1], pos[2],
              direction[0], direction[1], direction[2],
              color=color, alpha=alpha, arrow_length_ratio=0.3, lw=2)


def visualize_pose_3d(ax, pose, label='', scale=0.05):
    """Visualize a pose in 3D with coordinate frame."""
    pos = pose[:3]
    rot_vec = pose[3:]
    rot = R.from_rotvec(rot_vec)
    rot_mat = rot.as_matrix()
    
    # Draw coordinate frame axes
    colors = ['r', 'g', 'b']
    axis_labels = ['X', 'Y', 'Z']
    for i in range(3):
        direction = rot_mat[:, i] * scale
        draw_arrow_3d(ax, pos, direction, color=colors[i], length=scale, alpha=0.7)
        # Add axis label
        label_pos = pos + direction * 1.2
        ax.text(label_pos[0], label_pos[1], label_pos[2], 
                f'{axis_labels[i]}', fontsize=10, color=colors[i])
    
    # Draw position point
    ax.scatter([pos[0]], [pos[1]], [pos[2]], c='k', s=50, marker='o')
    if label:
        ax.text(pos[0], pos[1], pos[2], f'  {label}', fontsize=8)


def get_euler_xyz_from_pose(pose):
    """Convert pose rotation vector to Euler XYZ angles."""
    rot_vec = pose[3:]
    rot = R.from_rotvec(rot_vec)
    euler_xyz = rot.as_euler('XYZ', degrees=True)
    return euler_xyz


@click.command()
@click.option('--can_name', default='can0', help='CAN interface name (e.g., can0)')
@click.option('--test_movement', is_flag=True, default=False,
              help='Test small movements in each axis to verify orientation')
@click.option('--save_plot', type=str, default=None, help='Save plot to file')
def main(can_name, test_movement, save_plot):
    """Check Piper EE coordinate frame alignment."""
    
    print("=" * 80)
    print("Piper EE Coordinate Frame Check")
    print("=" * 80)
    
    # Initialize Piper interface
    print(f"\nConnecting to Piper robot on {can_name}...")
    piper = PiperInterface(can_name=can_name)
    
    try:
        # Get current EE pose
        print("\nGetting current EE pose from SDK...")
        pose = piper.get_ee_pose()
        
        print(f"\nCurrent EE Pose:")
        print(f"  Position (m): [{pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}]")
        print(f"  Rotation vector (rad): [{pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}]")
        
        # Convert to Euler XYZ for readability
        euler_xyz = get_euler_xyz_from_pose(pose)
        print(f"  Euler XYZ (deg): [{euler_xyz[0]:.2f}, {euler_xyz[1]:.2f}, {euler_xyz[2]:.2f}]")
        
        # Get rotation matrix to understand axis orientations
        rot_vec = pose[3:]
        rot = R.from_rotvec(rot_vec)
        rot_mat = rot.as_matrix()
        
        print(f"\nRotation Matrix (columns = X, Y, Z axes):")
        print(f"  X-axis (red):   [{rot_mat[0,0]:.3f}, {rot_mat[1,0]:.3f}, {rot_mat[2,0]:.3f}]")
        print(f"  Y-axis (green): [{rot_mat[0,1]:.3f}, {rot_mat[1,1]:.3f}, {rot_mat[2,1]:.3f}]")
        print(f"  Z-axis (blue):  [{rot_mat[0,2]:.3f}, {rot_mat[1,2]:.3f}, {rot_mat[2,2]:.3f}]")
        
        # Expected UMI convention: +Z forward, +Y up, +X right
        # In robot base frame (assuming standard right-handed):
        # - X: right (positive = rightward)
        # - Y: up (positive = upward)
        # - Z: forward (positive = forward)
        
        print(f"\n" + "=" * 80)
        print("Expected UMI Convention:")
        print("  Base frame: Right-handed")
        print("    +X: Right (positive = rightward)")
        print("    +Y: Up (positive = upward)")
        print("    +Z: Forward (positive = forward)")
        print("=" * 80)
        
        # Analyze current orientation
        print(f"\nCurrent EE Frame Analysis:")
        x_axis = rot_mat[:, 0]
        y_axis = rot_mat[:, 1]
        z_axis = rot_mat[:, 2]
        
        print(f"  X-axis direction: [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}]")
        print(f"  Y-axis direction: [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}]")
        print(f"  Z-axis direction: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
        
        # Check alignment with base frame axes
        base_x = np.array([1, 0, 0])  # Base X (right)
        base_y = np.array([0, 1, 0])  # Base Y (up)
        base_z = np.array([0, 0, 1])  # Base Z (forward)
        
        x_dot_base_x = np.dot(x_axis, base_x)
        x_dot_base_y = np.dot(x_axis, base_y)
        x_dot_base_z = np.dot(x_axis, base_z)
        
        y_dot_base_x = np.dot(y_axis, base_x)
        y_dot_base_y = np.dot(y_axis, base_y)
        y_dot_base_z = np.dot(y_axis, base_z)
        
        z_dot_base_x = np.dot(z_axis, base_x)
        z_dot_base_y = np.dot(z_axis, base_y)
        z_dot_base_z = np.dot(z_axis, base_z)
        
        print(f"\nEE Frame Alignment with Base Frame:")
        print(f"  EE X-axis alignment:")
        print(f"    with Base X (right): {x_dot_base_x:.3f}")
        print(f"    with Base Y (up):    {x_dot_base_y:.3f}")
        print(f"    with Base Z (forward): {x_dot_base_z:.3f}")
        print(f"  EE Y-axis alignment:")
        print(f"    with Base X (right): {y_dot_base_x:.3f}")
        print(f"    with Base Y (up):    {y_dot_base_y:.3f}")
        print(f"    with Base Z (forward): {y_dot_base_z:.3f}")
        print(f"  EE Z-axis alignment:")
        print(f"    with Base X (right): {z_dot_base_x:.3f}")
        print(f"    with Base Y (up):    {z_dot_base_y:.3f}")
        print(f"    with Base Z (forward): {z_dot_base_z:.3f}")
        
        # Visualize
        print(f"\nCreating visualization...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw base frame at origin
        base_origin = np.array([0, 0, 0])
        base_scale = 0.1
        draw_arrow_3d(ax, base_origin, np.array([base_scale, 0, 0]), 'r', base_scale, 0.5)
        draw_arrow_3d(ax, base_origin, np.array([0, base_scale, 0]), 'g', base_scale, 0.5)
        draw_arrow_3d(ax, base_origin, np.array([0, 0, base_scale]), 'b', base_scale, 0.5)
        ax.text(base_scale*1.2, 0, 0, 'Base X', fontsize=8, color='r')
        ax.text(0, base_scale*1.2, 0, 'Base Y', fontsize=8, color='g')
        ax.text(0, 0, base_scale*1.2, 'Base Z', fontsize=8, color='b')
        
        # Draw EE frame
        visualize_pose_3d(ax, pose, 'EE', scale=0.05)
        
        # Set equal aspect ratio
        max_range = max(abs(pose[0]), abs(pose[1]), abs(pose[2])) + 0.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Piper EE Coordinate Frame\n(Red=X, Green=Y, Blue=Z)')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=150)
            print(f"Plot saved to: {save_plot}")
        else:
            plt.show()
        
        # Test movement if requested
        if test_movement:
            print(f"\n" + "=" * 80)
            print("Testing small movements in each axis...")
            print("=" * 80)
            print("WARNING: Robot will move! Press Enter to continue or Ctrl+C to cancel.")
            input()
            
            movement_distance = 0.02  # 2 cm
            wait_time = 2.0  # seconds
            
            # Test X-axis movement
            print(f"\nMoving +{movement_distance}m in X-axis (should move right)...")
            test_pose = pose.copy()
            test_pose[0] += movement_distance
            piper.update_desired_ee_pose(test_pose)
            time.sleep(wait_time)
            
            # Return to original
            piper.update_desired_ee_pose(pose)
            time.sleep(wait_time)
            
            # Test Y-axis movement
            print(f"Moving +{movement_distance}m in Y-axis (should move up)...")
            test_pose = pose.copy()
            test_pose[1] += movement_distance
            piper.update_desired_ee_pose(test_pose)
            time.sleep(wait_time)
            
            # Return to original
            piper.update_desired_ee_pose(pose)
            time.sleep(wait_time)
            
            # Test Z-axis movement
            print(f"Moving +{movement_distance}m in Z-axis (should move forward)...")
            test_pose = pose.copy()
            test_pose[2] += movement_distance
            piper.update_desired_ee_pose(test_pose)
            time.sleep(wait_time)
            
            # Return to original
            print("Returning to original pose...")
            piper.update_desired_ee_pose(pose)
            time.sleep(wait_time)
            
            print("Movement test completed!")
        
    finally:
        piper.close()
        print("\nDisconnected from robot.")


if __name__ == '__main__':
    main()

