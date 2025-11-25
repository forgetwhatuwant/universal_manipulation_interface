#!/usr/bin/env python3
"""
Generate Piper EE trajectory from camera trajectory using delta-based approach.
Visualizes the resulting trajectory.

Usage:
    python scripts_real/generate_piper_ee_trajectory.py \
        --csv_path example_demo_session/demos/demo_*/camera_trajectory.csv \
        --start_pose_type zero  # or 'current' or 'custom'
        --visualize
"""

import sys
import os
import click
import numpy as np
import pandas as pd
import scipy.spatial.transform as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def quaternion_to_rotvec(qx, qy, qz, qw):
    """Convert quaternion [x, y, z, w] to rotation vector [rx, ry, rz]."""
    quat = np.array([qx, qy, qz, qw])
    rot = st.Rotation.from_quat(quat)
    return rot.as_rotvec()

def load_camera_trajectory(csv_path):
    """Load camera trajectory from CSV file."""
    print(f"Loading trajectory from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    positions = df[['x', 'y', 'z']].values
    quaternions = df[['q_x', 'q_y', 'q_z', 'q_w']].values
    timestamps = df['timestamp'].values
    
    rotation_vectors = np.array([
        quaternion_to_rotvec(q[0], q[1], q[2], q[3])
        for q in quaternions
    ])
    
    poses = np.hstack([positions, rotation_vectors])
    
    print(f"Loaded {len(poses)} poses")
    print(f"  Time range: {timestamps[0]:.3f} - {timestamps[-1]:.3f} seconds")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    
    return poses, timestamps

def get_piper_zero_pose():
    """Get Piper zero position EE pose (from piper_fk.py)."""
    # From piper_fk.py line 26: [56.128, 0.0, 213.266, 0.0, 85.0, 0.0] mm/deg
    pos_mm = np.array([56.128, 0.0, 213.266])
    pos_m = pos_mm / 1000.0
    
    rpy_deg = np.array([0.0, 85.0, 0.0])
    rpy_rad = np.radians(rpy_deg)
    rot = st.Rotation.from_euler('XYZ', rpy_rad)
    rot_vec = rot.as_rotvec()
    
    return np.concatenate([pos_m, rot_vec])

def get_piper_current_pose_from_sdk():
    """Get current pose from SDK (from user's provided values)."""
    # From user's SDK output:
    # X_axis: 57091 (0.001mm) = 57.091 mm = 0.057091 m
    # Y_axis: 0
    # Z_axis: 260008 (0.001mm) = 260.008 mm = 0.260008 m
    # RX_axis: 0 (0.001 deg) = 0 deg
    # RY_axis: 85040 (0.001 deg) = 85.04 deg
    # RZ_axis: 0 (0.001 deg) = 0 deg
    
    pos_mm = np.array([57.091, 0.0, 260.008])
    pos_m = pos_mm / 1000.0
    
    rpy_deg = np.array([0.0, 85.04, 0.0])
    rpy_rad = np.radians(rpy_deg)
    rot = st.Rotation.from_euler('XYZ', rpy_rad)
    rot_vec = rot.as_rotvec()
    
    return np.concatenate([pos_m, rot_vec])

@click.command()
@click.option('--csv_path', '-i', required=True, type=str, 
              help='Path to camera trajectory CSV file')
@click.option('--start_pose_type', default='zero', type=click.Choice(['zero', 'current', 'custom']),
              help='Starting pose: zero (Piper FK zero), current (from SDK), or custom')
@click.option('--start_x', default=0.0, type=float, help='Custom start X (m)')
@click.option('--start_y', default=0.0, type=float, help='Custom start Y (m)')
@click.option('--start_z', default=0.0, type=float, help='Custom start Z (m)')
@click.option('--start_rx', default=0.0, type=float, help='Custom start RX (rad)')
@click.option('--start_ry', default=0.0, type=float, help='Custom start RY (rad)')
@click.option('--start_rz', default=0.0, type=float, help='Custom start RZ (rad)')
@click.option('--output', '-o', type=str, default=None, help='Save trajectory to CSV file')
@click.option('--visualize', is_flag=True, default=True, help='Visualize trajectory')
@click.option('--save_plot', type=str, default=None, help='Save plot to file')
def main(csv_path, start_pose_type, start_x, start_y, start_z, start_rx, start_ry, start_rz,
         output, visualize, save_plot):
    """Generate Piper EE trajectory from camera trajectory using delta-based approach."""
    
    # Load camera trajectory
    poses, timestamps = load_camera_trajectory(csv_path)
    
    # Determine starting pose
    if start_pose_type == 'zero':
        start_pose = get_piper_zero_pose()
        print(f"\nUsing Piper ZERO position as starting pose:")
    elif start_pose_type == 'current':
        start_pose = get_piper_current_pose_from_sdk()
        print(f"\nUsing Piper CURRENT position (from SDK) as starting pose:")
    else:  # custom
        start_pose = np.array([start_x, start_y, start_z, start_rx, start_ry, start_rz])
        print(f"\nUsing CUSTOM starting pose:")
    
    print(f"  Position: [{start_pose[0]:.4f}, {start_pose[1]:.4f}, {start_pose[2]:.4f}] m")
    print(f"  Rotation: [{start_pose[3]:.4f}, {start_pose[4]:.4f}, {start_pose[5]:.4f}] rad")
    
    # Calculate deltas relative to first frame
    print(f"\nCalculating deltas relative to first camera frame...")
    deltas = np.zeros_like(poses)
    deltas[0] = np.zeros(6)  # First frame: no movement
    
    r_traj_first = st.Rotation.from_rotvec(poses[0, 3:])
    
    for i in range(1, len(poses)):
        # Position delta (in camera frame)
        deltas[i, :3] = poses[i, :3] - poses[0, :3]
        
        # Rotation delta
        r_traj_i = st.Rotation.from_rotvec(poses[i, 3:])
        r_delta = r_traj_i * r_traj_first.inv()
        deltas[i, 3:] = r_delta.as_rotvec()
    
    # Transform camera frame deltas to robot base frame
    # Camera frame: Z-forward, X-right, Y-down
    # Robot frame: X-forward, Y-left/right, Z-up
    print(f"\nTransforming deltas from camera frame to robot base frame...")
    print(f"  Camera frame: Z-forward, X-right, Y-down")
    print(f"  Robot frame: X-forward, Y-left/right, Z-up")
    
    camera_to_robot_transform = np.array([
        [0, 0, 1],   # Camera Z (forward) → Robot X (forward)
        [1, 0, 0],   # Camera X (right) → Robot Y (sideways)
        [0, -1, 0],  # Camera Y (down) → Robot Z (up, negative because camera Y is down)
    ])
    
    # Transform position deltas
    for i in range(len(deltas)):
        deltas[i, :3] = camera_to_robot_transform @ deltas[i, :3]
    
    # Transform rotation deltas (rotation vectors need to be converted to rotation matrices first)
    for i in range(len(deltas)):
        if np.linalg.norm(deltas[i, 3:]) > 1e-6:  # Only transform non-zero rotations
            # Convert rotation vector to rotation matrix
            r_delta_cam = st.Rotation.from_rotvec(deltas[i, 3:])
            r_mat_cam = r_delta_cam.as_matrix()
            
            # Transform rotation matrix: R_robot = T @ R_cam @ T^T
            r_mat_robot = camera_to_robot_transform @ r_mat_cam @ camera_to_robot_transform.T
            
            # Convert back to rotation vector
            r_delta_robot = st.Rotation.from_matrix(r_mat_robot)
            deltas[i, 3:] = r_delta_robot.as_rotvec()
    
    max_pos_delta = np.max(np.linalg.norm(deltas[:, :3], axis=1))
    max_rot_delta = np.max(np.linalg.norm(deltas[:, 3:], axis=1))
    print(f"  Max position delta (after transform): {max_pos_delta:.4f} m")
    print(f"  Max rotation delta (after transform): {np.degrees(max_rot_delta):.2f} deg")
    
    # Apply deltas to starting pose to generate EE trajectory
    print(f"\nGenerating Piper EE trajectory...")
    
    # Transform camera's first frame rotation to robot frame and align with robot's initial rotation
    r_cam_first = st.Rotation.from_rotvec(poses[0, 3:])
    r_cam_first_mat = r_cam_first.as_matrix()
    # Transform rotation matrix: R_robot = T @ R_cam @ T^T
    r_cam_first_robot_mat = camera_to_robot_transform @ r_cam_first_mat @ camera_to_robot_transform.T
    r_cam_first_robot = st.Rotation.from_matrix(r_cam_first_robot_mat)
    
    # Calculate rotation offset: what rotation takes us from transformed camera first frame to robot start
    r_start = st.Rotation.from_rotvec(start_pose[3:])
    r_align = r_start * r_cam_first_robot.inv()  # Rotation to align camera frame to robot frame
    
    print(f"  Camera first frame rotation (transformed to robot frame): {r_cam_first_robot.as_rotvec()}")
    print(f"  Robot start rotation: {r_start.as_rotvec()}")
    print(f"  Alignment rotation: {r_align.as_rotvec()}")
    
    ee_trajectory = np.zeros_like(poses)
    
    for i in range(len(poses)):
        # Position: start + delta
        ee_trajectory[i, :3] = start_pose[:3] + deltas[i, :3]
        
        # Rotation: apply transformed delta, then align with robot's initial orientation
        r_delta = st.Rotation.from_rotvec(deltas[i, 3:])
        # First apply the delta relative to transformed camera first frame
        r_after_delta = r_delta * r_cam_first_robot
        # Then apply alignment to match robot's initial orientation
        r_target = r_align * r_after_delta
        ee_trajectory[i, 3:] = r_target.as_rotvec()
    
    print(f"Generated {len(ee_trajectory)} EE poses")
    
    # Calculate trajectory bounds
    pos_min = np.min(ee_trajectory[:, :3], axis=0)
    pos_max = np.max(ee_trajectory[:, :3], axis=0)
    print(f"\nTrajectory bounds:")
    print(f"  X: [{pos_min[0]:.4f}, {pos_max[0]:.4f}] m")
    print(f"  Y: [{pos_min[1]:.4f}, {pos_max[1]:.4f}] m")
    print(f"  Z: [{pos_min[2]:.4f}, {pos_max[2]:.4f}] m")
    
    # Save to CSV if requested
    if output:
        print(f"\nSaving trajectory to: {output}")
        df_out = pd.DataFrame({
            'timestamp': timestamps,
            'x': ee_trajectory[:, 0],
            'y': ee_trajectory[:, 1],
            'z': ee_trajectory[:, 2],
            'rx': ee_trajectory[:, 3],
            'ry': ee_trajectory[:, 4],
            'rz': ee_trajectory[:, 5],
        })
        df_out.to_csv(output, index=False)
        print("Saved!")
    
    # Visualize
    if visualize:
        print(f"\nVisualizing trajectory...")
        
        fig = plt.figure(figsize=(18, 12))
        plt.suptitle(f"Piper EE Trajectory Generated from Camera Trajectory\n{os.path.basename(csv_path)}", fontsize=16)
        
        # 1. 3D trajectory
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2], 
                'b-', label='EE Trajectory', linewidth=2, alpha=0.7)
        ax1.scatter([start_pose[0]], [start_pose[1]], [start_pose[2]], 
                   c='g', s=200, marker='o', label='Start', zorder=5)
        ax1.scatter([ee_trajectory[-1, 0]], [ee_trajectory[-1, 1]], [ee_trajectory[-1, 2]], 
                   c='r', s=200, marker='s', label='End', zorder=5)
        
        # Draw coordinate frames at key points
        from scipy.spatial.transform import Rotation as R
        for idx in [0, len(ee_trajectory)//2, len(ee_trajectory)-1]:
            pos = ee_trajectory[idx, :3]
            rot_vec = ee_trajectory[idx, 3:]
            rot = R.from_rotvec(rot_vec)
            rot_mat = rot.as_matrix()
            scale = 0.02
            colors = ['r', 'g', 'b']
            for i in range(3):
                direction = rot_mat[:, i] * scale
                ax1.quiver(pos[0], pos[1], pos[2],
                          direction[0], direction[1], direction[2],
                          color=colors[i], alpha=0.7, arrow_length_ratio=0.3, lw=2)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D EE Trajectory')
        ax1.legend()
        
        # 2. Position over time
        ax2 = fig.add_subplot(232)
        t_norm = timestamps - timestamps[0]
        ax2.plot(t_norm, ee_trajectory[:, 0], 'r-', label='X', linewidth=2)
        ax2.plot(t_norm, ee_trajectory[:, 1], 'g-', label='Y', linewidth=2)
        ax2.plot(t_norm, ee_trajectory[:, 2], 'b-', label='Z', linewidth=2)
        ax2.set_title('Position over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rotation over time
        ax3 = fig.add_subplot(233)
        ax3.plot(t_norm, ee_trajectory[:, 3], 'r-', label='Rx', linewidth=2)
        ax3.plot(t_norm, ee_trajectory[:, 4], 'g-', label='Ry', linewidth=2)
        ax3.plot(t_norm, ee_trajectory[:, 5], 'b-', label='Rz', linewidth=2)
        ax3.set_title('Rotation over Time (Rotation Vector)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Rotation (rad)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Compare with original camera trajectory
        ax4 = fig.add_subplot(234, projection='3d')
        ax4.plot(poses[:, 0], poses[:, 1], poses[:, 2], 
                'r--', label='Camera Traj', alpha=0.5, linewidth=1)
        ax4.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2], 
                'b-', label='EE Traj', alpha=0.7, linewidth=2)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('Camera vs EE Trajectory (Shape Comparison)')
        ax4.legend()
        
        # 5. XY top view
        ax5 = fig.add_subplot(235)
        ax5.plot(ee_trajectory[:, 0], ee_trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax5.scatter([start_pose[0]], [start_pose[1]], c='g', s=200, marker='o', label='Start', zorder=5)
        ax5.scatter([ee_trajectory[-1, 0]], [ee_trajectory[-1, 1]], c='r', s=200, marker='s', label='End', zorder=5)
        ax5.set_title('XY Top View')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # 6. XZ side view
        ax6 = fig.add_subplot(236)
        ax6.plot(ee_trajectory[:, 0], ee_trajectory[:, 2], 'b-', linewidth=2, alpha=0.7)
        ax6.scatter([start_pose[0]], [start_pose[2]], c='g', s=200, marker='o', label='Start', zorder=5)
        ax6.scatter([ee_trajectory[-1, 0]], [ee_trajectory[-1, 2]], c='r', s=200, marker='s', label='End', zorder=5)
        ax6.set_title('XZ Side View')
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Z (m)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_aspect('equal')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_plot:
            plt.savefig(save_plot, dpi=150)
            print(f"Plot saved to: {save_plot}")
        
        plt.show()
    
    print("\nDone!")

if __name__ == '__main__':
    main()