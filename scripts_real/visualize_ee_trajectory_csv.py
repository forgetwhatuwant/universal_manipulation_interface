#!/usr/bin/env python3
"""
Visualize EE trajectory from CSV file.

Usage:
    python scripts_real/visualize_ee_trajectory_csv.py --input piper_ee_trajectory.csv
"""

import sys
import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def visualize_pose_3d(ax, pose, scale=0.02):
    """Visualize a pose in 3D with coordinate frame."""
    pos = pose[:3]
    rot_vec = pose[3:]
    
    rot = R.from_rotvec(rot_vec)
    rot_mat = rot.as_matrix()
    
    colors = ['r', 'g', 'b']
    for i in range(3):
        direction = rot_mat[:, i] * scale
        ax.quiver(pos[0], pos[1], pos[2],
                  direction[0], direction[1], direction[2],
                  color=colors[i], alpha=0.7, arrow_length_ratio=0.3, lw=2)

@click.command()
@click.option('--input', '-i', required=True, type=str, help='Path to CSV trajectory file')
@click.option('--save_plot', type=str, default=None, help='Save plot to file')
@click.option('--show', is_flag=True, default=True, help='Show plot interactively')
def main(input, save_plot, show):
    """Visualize EE trajectory from CSV."""
    
    print(f"Loading trajectory from: {input}")
    df = pd.read_csv(input)
    
    timestamps = df['timestamp'].values
    poses = df[['x', 'y', 'z', 'rx', 'ry', 'rz']].values
    
    print(f"Loaded {len(poses)} poses")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    
    # Normalize timestamps
    t_norm = timestamps - timestamps[0]
    
    # Create plots
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"Piper EE Trajectory Visualization\n{os.path.basename(input)}", fontsize=16)
    
    # 1. 3D trajectory
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(poses[:, 0], poses[:, 1], poses[:, 2], 
            'b-', label='EE Trajectory', linewidth=2, alpha=0.7)
    ax1.scatter([poses[0, 0]], [poses[0, 1]], [poses[0, 2]], 
               c='g', s=200, marker='o', label='Start', zorder=5)
    ax1.scatter([poses[-1, 0]], [poses[-1, 1]], [poses[-1, 2]], 
               c='r', s=200, marker='s', label='End', zorder=5)
    
    # Draw coordinate frames at key points
    for idx in [0, len(poses)//2, len(poses)-1]:
        visualize_pose_3d(ax1, poses[idx], scale=0.02)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D EE Trajectory')
    ax1.legend()
    
    # 2. Position over time
    ax2 = fig.add_subplot(232)
    ax2.plot(t_norm, poses[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(t_norm, poses[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(t_norm, poses[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_title('Position over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rotation over time
    ax3 = fig.add_subplot(233)
    ax3.plot(t_norm, poses[:, 3], 'r-', label='Rx', linewidth=2)
    ax3.plot(t_norm, poses[:, 4], 'g-', label='Ry', linewidth=2)
    ax3.plot(t_norm, poses[:, 5], 'b-', label='Rz', linewidth=2)
    ax3.set_title('Rotation over Time (Rotation Vector)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rotation (rad)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Position velocity
    ax4 = fig.add_subplot(234)
    pos_vel = np.diff(poses[:, :3], axis=0) / np.diff(t_norm).reshape(-1, 1)
    t_vel = (t_norm[1:] + t_norm[:-1]) / 2
    speeds = np.linalg.norm(pos_vel, axis=1)
    ax4.plot(t_vel, speeds, 'k-', linewidth=2)
    ax4.set_title('Position Speed over Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(np.mean(speeds), color='r', linestyle='--', 
                label=f'Mean: {np.mean(speeds):.3f} m/s')
    ax4.legend()
    
    # 5. XY top view
    ax5 = fig.add_subplot(235)
    ax5.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax5.scatter([poses[0, 0]], [poses[0, 1]], c='g', s=200, marker='o', label='Start', zorder=5)
    ax5.scatter([poses[-1, 0]], [poses[-1, 1]], c='r', s=200, marker='s', label='End', zorder=5)
    ax5.set_title('XY Top View')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. XZ side view
    ax6 = fig.add_subplot(236)
    ax6.plot(poses[:, 0], poses[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax6.scatter([poses[0, 0]], [poses[0, 2]], c='g', s=200, marker='o', label='Start', zorder=5)
    ax6.scatter([poses[-1, 0]], [poses[-1, 2]], c='r', s=200, marker='s', label='End', zorder=5)
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
    
    print("\nDone!")

if __name__ == '__main__':
    main()