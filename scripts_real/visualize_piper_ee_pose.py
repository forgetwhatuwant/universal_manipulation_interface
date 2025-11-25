#!/usr/bin/env python3
"""
Visualize Piper robot end-effector pose (position and orientation).

Can visualize:
1. Live EE pose from robot
2. EE pose from recorded replay pickle file
3. Joint positions and FK-calculated poses

Usage:
    # Live visualization
    python scripts_real/visualize_piper_ee_pose.py --can_name can0 --live
    
    # From replay file
    python scripts_real/visualize_piper_ee_pose.py --input my_replay.pkl
"""

import sys
import os
import time
import click
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def draw_arrow_3d(ax, pos, direction, color='r', length=0.05, alpha=0.8):
    """Draw a 3D arrow representing orientation using quiver."""
    # Normalize direction
    direction = direction / np.linalg.norm(direction) * length
    
    # Use quiver for 3D arrows
    ax.quiver(pos[0], pos[1], pos[2],
              direction[0], direction[1], direction[2],
              color=color, alpha=alpha, arrow_length_ratio=0.3, lw=2)

def visualize_pose_3d(ax, pose, label='', scale=0.05):
    """Visualize a pose in 3D with coordinate frame."""
    pos = pose[:3]
    rot_vec = pose[3:]
    
    # Convert rotation vector to rotation matrix
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(rot_vec)
    rot_mat = rot.as_matrix()
    
    # Draw coordinate frame (X=red, Y=green, Z=blue)
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        direction = rot_mat[:, i] * scale
        draw_arrow_3d(ax, pos, direction, color=colors[i], length=scale, alpha=0.7)
    
    # Draw position point
    ax.scatter([pos[0]], [pos[1]], [pos[2]], c='k', s=50, marker='o')
    
    if label:
        ax.text(pos[0], pos[1], pos[2], f'  {label}', fontsize=8)

@click.command()
@click.option('--can_name', default='can0', help='CAN interface name (for live mode)')
@click.option('--live', is_flag=True, default=False, help='Visualize live EE pose from robot')
@click.option('--input', '-i', type=str, default=None, help='Path to replay pickle file')
@click.option('--duration', default=10.0, type=float, help='Duration for live visualization (seconds)')
@click.option('--update_rate', default=10.0, type=float, help='Update rate for live visualization (Hz)')
@click.option('--save_plot', type=str, default=None, help='Save plot to file')
@click.option('--show', is_flag=True, default=True, help='Show plot interactively')
def main(can_name, live, input, duration, update_rate, save_plot, show):
    """Visualize Piper robot end-effector pose."""
    
    if live and input:
        print("Error: Cannot use both --live and --input. Choose one.")
        return
    
    if not live and not input:
        print("Error: Must specify either --live or --input")
        return
    
    if live:
        # Live visualization
        print(f"Connecting to Piper robot on {can_name}...")
        from umi.real_world.piper_interface import PiperInterface
        
        robot = PiperInterface(can_name=can_name)
        
        print(f"Visualizing live EE pose for {duration} seconds at {update_rate} Hz...")
        print("Close plot window or press Ctrl+C to stop")
        
        poses = []
        timestamps = []
        joint_positions = []
        
        dt = 1.0 / update_rate
        t_start = time.time()
        
        fig = plt.figure(figsize=(16, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.set_title('Live EE Position Trajectory')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Position over time
        ax2 = fig.add_subplot(232)
        ax2.set_title('Position over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.grid(True, alpha=0.3)
        
        # Rotation over time
        ax3 = fig.add_subplot(233)
        ax3.set_title('Rotation over Time (Rotation Vector)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Rotation (rad)')
        ax3.grid(True, alpha=0.3)
        
        # Joint positions
        ax4 = fig.add_subplot(234)
        ax4.set_title('Joint Positions')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angle (rad)')
        ax4.grid(True, alpha=0.3)
        
        # XY top view
        ax5 = fig.add_subplot(235)
        ax5.set_title('XY Top View')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # XZ side view
        ax6 = fig.add_subplot(236)
        ax6.set_title('XZ Side View')
        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Z (m)')
        ax6.grid(True, alpha=0.3)
        ax6.set_aspect('equal')
        
        plt.ion()  # Interactive mode
        plt.show()
        
        try:
            while time.time() - t_start < duration:
                # Get current pose
                ee_pose = robot.get_ee_pose()
                joints = robot.get_joint_positions()
                
                t = time.time() - t_start
                poses.append(ee_pose.copy())
                timestamps.append(t)
                joint_positions.append(joints.copy())
                
                if len(poses) > 1:
                    # Update plots
                    poses_arr = np.array(poses)
                    joints_arr = np.array(joint_positions)
                    times_arr = np.array(timestamps)
                    
                    # Clear and redraw
                    ax1.clear()
                    ax1.plot(poses_arr[:, 0], poses_arr[:, 1], poses_arr[:, 2], 'b-', alpha=0.6)
                    ax1.scatter([poses_arr[-1, 0]], [poses_arr[-1, 1]], [poses_arr[-1, 2]], 
                               c='r', s=100, marker='o', label='Current')
                    visualize_pose_3d(ax1, poses_arr[-1], scale=0.03)
                    ax1.set_title(f'Live EE Trajectory (t={t:.1f}s)')
                    ax1.set_xlabel('X (m)')
                    ax1.set_ylabel('Y (m)')
                    ax1.set_zlabel('Z (m)')
                    ax1.legend()
                    
                    ax2.clear()
                    ax2.plot(times_arr, poses_arr[:, 0], 'r-', label='X')
                    ax2.plot(times_arr, poses_arr[:, 1], 'g-', label='Y')
                    ax2.plot(times_arr, poses_arr[:, 2], 'b-', label='Z')
                    ax2.set_title('Position over Time')
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('Position (m)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    ax3.clear()
                    ax3.plot(times_arr, poses_arr[:, 3], 'r-', label='Rx')
                    ax3.plot(times_arr, poses_arr[:, 4], 'g-', label='Ry')
                    ax3.plot(times_arr, poses_arr[:, 5], 'b-', label='Rz')
                    ax3.set_title('Rotation over Time')
                    ax3.set_xlabel('Time (s)')
                    ax3.set_ylabel('Rotation (rad)')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    
                    ax4.clear()
                    for i in range(6):
                        ax4.plot(times_arr, joints_arr[:, i], label=f'J{i+1}')
                    ax4.set_title('Joint Positions')
                    ax4.set_xlabel('Time (s)')
                    ax4.set_ylabel('Angle (rad)')
                    ax4.legend(fontsize='small', ncol=2)
                    ax4.grid(True, alpha=0.3)
                    
                    ax5.clear()
                    ax5.plot(poses_arr[:, 0], poses_arr[:, 1], 'b-', alpha=0.6)
                    ax5.scatter([poses_arr[-1, 0]], [poses_arr[-1, 1]], c='r', s=100)
                    ax5.set_title('XY Top View')
                    ax5.set_xlabel('X (m)')
                    ax5.set_ylabel('Y (m)')
                    ax5.grid(True, alpha=0.3)
                    ax5.set_aspect('equal')
                    
                    ax6.clear()
                    ax6.plot(poses_arr[:, 0], poses_arr[:, 2], 'b-', alpha=0.6)
                    ax6.scatter([poses_arr[-1, 0]], [poses_arr[-1, 2]], c='r', s=100)
                    ax6.set_title('XZ Side View')
                    ax6.set_xlabel('X (m)')
                    ax6.set_ylabel('Z (m)')
                    ax6.grid(True, alpha=0.3)
                    ax6.set_aspect('equal')
                    
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.01)
                
                time.sleep(dt)
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            robot.close()
            plt.ioff()
    
    else:
        # Load from pickle file
        import pickle
        
        print(f"Loading data from: {input}")
        with open(input, 'rb') as f:
            data = pickle.load(f)
        
        actual_poses = data.get('actual_poses')
        actual_timestamps = data.get('actual_timestamps')
        actual_q = data.get('actual_q')
        target_poses = data.get('target_poses')
        target_timestamps = data.get('target_timestamps')
        
        if actual_poses is None or len(actual_poses) == 0:
            print("Error: No actual pose data found in file.")
            return
        
        # Normalize timestamps
        t0 = actual_timestamps[0]
        actual_t = actual_timestamps - t0
        
        if target_poses is not None:
            target_t0 = target_timestamps[0]
            target_t = target_timestamps - target_t0
        
        # Create plots
        fig = plt.figure(figsize=(18, 12))
        plt.suptitle(f"Piper EE Pose Visualization\n{os.path.basename(input)}", fontsize=16)
        
        # 1. 3D trajectory
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(actual_poses[:, 0], actual_poses[:, 1], actual_poses[:, 2], 
                'r-', label='Actual', alpha=0.7, linewidth=2)
        if target_poses is not None:
            ax1.plot(target_poses[:, 0], target_poses[:, 1], target_poses[:, 2], 
                    'b--', label='Target', alpha=0.5)
        
        # Draw coordinate frames at start, middle, end
        for idx in [0, len(actual_poses)//2, len(actual_poses)-1]:
            visualize_pose_3d(ax1, actual_poses[idx], 
                             label=f'{idx}', scale=0.02)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # 2. Position over time
        ax2 = fig.add_subplot(232)
        ax2.plot(actual_t, actual_poses[:, 0], 'r-', label='X', linewidth=2)
        ax2.plot(actual_t, actual_poses[:, 1], 'g-', label='Y', linewidth=2)
        ax2.plot(actual_t, actual_poses[:, 2], 'b-', label='Z', linewidth=2)
        if target_poses is not None:
            ax2.plot(target_t, target_poses[:, 0], 'r--', alpha=0.5)
            ax2.plot(target_t, target_poses[:, 1], 'g--', alpha=0.5)
            ax2.plot(target_t, target_poses[:, 2], 'b--', alpha=0.5)
        ax2.set_title('Position over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rotation over time
        ax3 = fig.add_subplot(233)
        ax3.plot(actual_t, actual_poses[:, 3], 'r-', label='Rx', linewidth=2)
        ax3.plot(actual_t, actual_poses[:, 4], 'g-', label='Ry', linewidth=2)
        ax3.plot(actual_t, actual_poses[:, 5], 'b-', label='Rz', linewidth=2)
        if target_poses is not None:
            ax3.plot(target_t, target_poses[:, 3], 'r--', alpha=0.5)
            ax3.plot(target_t, target_poses[:, 4], 'g--', alpha=0.5)
            ax3.plot(target_t, target_poses[:, 5], 'b--', alpha=0.5)
        ax3.set_title('Rotation over Time (Rotation Vector)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Rotation (rad)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Joint positions
        if actual_q is not None:
            ax4 = fig.add_subplot(234)
            for i in range(actual_q.shape[1]):
                ax4.plot(actual_t, actual_q[:, i], label=f'J{i+1}')
            ax4.set_title('Joint Positions')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Angle (rad)')
            ax4.legend(fontsize='small', ncol=2)
            ax4.grid(True, alpha=0.3)
        
        # 5. XY top view
        ax5 = fig.add_subplot(235)
        ax5.plot(actual_poses[:, 0], actual_poses[:, 1], 'r-', label='Actual', linewidth=2)
        if target_poses is not None:
            ax5.plot(target_poses[:, 0], target_poses[:, 1], 'b--', label='Target', alpha=0.5)
        ax5.scatter([actual_poses[0, 0]], [actual_poses[0, 1]], c='g', s=100, marker='o', label='Start')
        ax5.scatter([actual_poses[-1, 0]], [actual_poses[-1, 1]], c='r', s=100, marker='s', label='End')
        ax5.set_title('XY Top View')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # 6. XZ side view
        ax6 = fig.add_subplot(236)
        ax6.plot(actual_poses[:, 0], actual_poses[:, 2], 'r-', label='Actual', linewidth=2)
        if target_poses is not None:
            ax6.plot(target_poses[:, 0], target_poses[:, 2], 'b--', label='Target', alpha=0.5)
        ax6.scatter([actual_poses[0, 0]], [actual_poses[0, 2]], c='g', s=100, marker='o', label='Start')
        ax6.scatter([actual_poses[-1, 0]], [actual_poses[-1, 2]], c='r', s=100, marker='s', label='End')
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
    
    if show:
        plt.show()

if __name__ == '__main__':
    main()

