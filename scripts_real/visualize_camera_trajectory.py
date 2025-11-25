#!/usr/bin/env python3
"""
Visualize camera trajectory from CSV file.

This script loads a camera trajectory CSV file and visualizes:
- 3D trajectory path
- Position and rotation over time
- Velocity and acceleration analysis
- Trajectory statistics

CSV Format Expected:
    frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w

Usage:
    python scripts_real/visualize_camera_trajectory.py \
        --csv_path example_demo_session/demos/demo_*/camera_trajectory.csv
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Check if conda environment is activated
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', '')
if CONDA_ENV != 'umi':
    print("=" * 80)
    print("WARNING: Conda environment 'umi' is not activated!")
    print(f"Current environment: {CONDA_ENV if CONDA_ENV else 'None'}")
    print("Please activate the umi environment:")
    print("  conda activate umi")
    print("=" * 80)
    # Don't exit, just warn - user might be using a different setup

import click
import numpy as np
import pandas as pd
import scipy.spatial.transform as st

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError as e:
    HAS_MATPLOTLIB = False
    print(f"Warning: matplotlib not available ({e})")
    print("Install matplotlib to enable visualization: pip install matplotlib")

def quaternion_to_rotvec(qx, qy, qz, qw):
    """
    Convert quaternion [x, y, z, w] to rotation vector [rx, ry, rz].
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        np.array([rx, ry, rz]): Rotation vector in radians
    """
    quat = np.array([qx, qy, qz, qw])
    rot = st.Rotation.from_quat(quat)
    return rot.as_rotvec()

def load_camera_trajectory(csv_path):
    """
    Load camera trajectory from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        tuple: (poses, timestamps, df) where poses is (N, 6) array [x,y,z,rx,ry,rz]
               and timestamps is (N,) array in seconds
    """
    print(f"Loading trajectory from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract position and quaternion
    positions = df[['x', 'y', 'z']].values  # (N, 3)
    quaternions = df[['q_x', 'q_y', 'q_z', 'q_w']].values  # (N, 4)
    timestamps = df['timestamp'].values  # (N,)
    
    # Convert quaternions to rotation vectors
    print("Converting quaternions to rotation vectors...")
    rotation_vectors = np.array([
        quaternion_to_rotvec(q[0], q[1], q[2], q[3])
        for q in quaternions
    ])  # (N, 3)
    
    # Combine position and rotation into pose format [x, y, z, rx, ry, rz]
    poses = np.hstack([positions, rotation_vectors])  # (N, 6)
    
    return poses, timestamps, df

def compute_velocities(positions, timestamps):
    """Compute velocities from positions and timestamps."""
    dt = np.diff(timestamps)
    dt = np.concatenate([[dt[0]], dt])  # Use first dt for first point
    velocities = np.diff(positions, axis=0) / dt[1:, np.newaxis]
    # Pad first velocity with zeros
    velocities = np.vstack([np.zeros((1, 3)), velocities])
    return velocities

def compute_angular_velocities(rotations, timestamps):
    """Compute angular velocities from rotation vectors and timestamps."""
    dt = np.diff(timestamps)
    dt = np.concatenate([[dt[0]], dt])
    
    # Convert rotation vectors to rotation objects
    rots = st.Rotation.from_rotvec(rotations)
    
    # Compute angular velocities
    angular_vels = np.zeros_like(rotations)
    for i in range(1, len(rots)):
        # Relative rotation
        rel_rot = rots[i] * rots[i-1].inv()
        # Angular velocity (rotation vector / dt)
        angular_vels[i] = rel_rot.as_rotvec() / dt[i]
    
    return angular_vels

@click.command()
@click.option('--csv_path', '-i', required=True, type=str,
              help='Path to camera trajectory CSV file')
@click.option('--output', '-o', type=str, default=None,
              help='Output path for saved figure (e.g., trajectory_vis.png). If not specified and CSV path provided, saves as camera_traj.png in same directory as CSV.')
@click.option('--show', is_flag=True, default=None,
              help='Show interactive plot window (default: True if no --output, False if --output specified)')
@click.option('--arrow_scale', default=0.02, type=float,
              help='Scale factor for orientation arrows in 3D plot')
@click.option('--arrow_skip', default=10, type=int,
              help='Show orientation arrow every N frames')
def main(csv_path, output, show, arrow_scale, arrow_skip):
    """
    Visualize camera trajectory.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization.")
        print("Please install it: pip install matplotlib")
        return
    
    # Auto-generate output path if not specified
    if output is None:
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        output = os.path.join(csv_dir, 'camera_traj.png')
        print(f"Auto-generated output path: {output}")
    
    # Load trajectory
    poses, timestamps, df = load_camera_trajectory(csv_path)
    
    positions = poses[:, :3]
    rotations = poses[:, 3:]
    
    # Compute velocities
    velocities = compute_velocities(positions, timestamps)
    angular_velocities = compute_angular_velocities(rotations, timestamps)
    
    # Compute speeds
    pos_speeds = np.linalg.norm(velocities, axis=1)
    rot_speeds = np.linalg.norm(angular_velocities, axis=1)
    
    # Statistics
    duration = timestamps[-1] - timestamps[0]
    avg_fps = 1.0 / np.mean(np.diff(timestamps))
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    max_pos_speed = np.max(pos_speeds)
    max_rot_speed = np.max(rot_speeds)
    avg_pos_speed = np.mean(pos_speeds)
    avg_rot_speed = np.mean(rot_speeds)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Trajectory Statistics")
    print("=" * 80)
    print(f"Total frames: {len(poses)}")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Total distance traveled: {total_distance:.4f} m")
    print(f"\nPosition:")
    print(f"  Range X: [{np.min(positions[:, 0]):.4f}, {np.max(positions[:, 0]):.4f}] m")
    print(f"  Range Y: [{np.min(positions[:, 1]):.4f}, {np.max(positions[:, 1]):.4f}] m")
    print(f"  Range Z: [{np.min(positions[:, 2]):.4f}, {np.max(positions[:, 2]):.4f}] m")
    print(f"  Max speed: {max_pos_speed:.4f} m/s")
    print(f"  Avg speed: {avg_pos_speed:.4f} m/s")
    print(f"\nRotation:")
    print(f"  Max angular speed: {max_rot_speed:.4f} rad/s ({np.degrees(max_rot_speed):.2f} deg/s)")
    print(f"  Avg angular speed: {avg_rot_speed:.4f} rad/s ({np.degrees(avg_rot_speed):.2f} deg/s)")
    print("=" * 80)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', linewidth=1, alpha=0.6, label='Trajectory')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                c='red', s=100, marker='s', label='End', zorder=5)
    
    # Add orientation arrows
    rots = st.Rotation.from_rotvec(rotations)
    for i in range(0, len(positions), arrow_skip):
        pos = positions[i]
        rot = rots[i]
        # Get forward direction (Z-axis in camera frame)
        forward = rot.apply([0, 0, 1])
        ax1.quiver(pos[0], pos[1], pos[2],
                  forward[0] * arrow_scale, forward[1] * arrow_scale, forward[2] * arrow_scale,
                  color='red', alpha=0.5, arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Path')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Position over time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(timestamps, positions[:, 0], 'r-', label='X', linewidth=1.5)
    ax2.plot(timestamps, positions[:, 1], 'g-', label='Y', linewidth=1.5)
    ax2.plot(timestamps, positions[:, 2], 'b-', label='Z', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rotation over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(timestamps, rotations[:, 0], 'r-', label='RX', linewidth=1.5)
    ax3.plot(timestamps, rotations[:, 1], 'g-', label='RY', linewidth=1.5)
    ax3.plot(timestamps, rotations[:, 2], 'b-', label='RZ', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Rotation Vector (rad)')
    ax3.set_title('Rotation vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Position speed over time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(timestamps, pos_speeds, 'b-', linewidth=1.5)
    ax4.axhline(y=max_pos_speed, color='r', linestyle='--', 
                label=f'Max: {max_pos_speed:.4f} m/s')
    ax4.axhline(y=avg_pos_speed, color='g', linestyle='--', 
                label=f'Avg: {avg_pos_speed:.4f} m/s')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Position Speed vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Angular speed over time
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(timestamps, rot_speeds, 'r-', linewidth=1.5)
    ax5.axhline(y=max_rot_speed, color='r', linestyle='--', 
                label=f'Max: {max_rot_speed:.4f} rad/s')
    ax5.axhline(y=avg_rot_speed, color='g', linestyle='--', 
                label=f'Avg: {avg_rot_speed:.4f} rad/s')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Speed (rad/s)')
    ax5.set_title('Angular Speed vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. XY top view
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, alpha=0.6)
    ax6.scatter(positions[0, 0], positions[0, 1],
                c='green', s=100, marker='o', label='Start', zorder=5)
    ax6.scatter(positions[-1, 0], positions[-1, 1],
                c='red', s=100, marker='s', label='End', zorder=5)
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('Top View (XY plane)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Determine if we should show the plot
    # Default: show if no output specified, don't show if output specified
    if show is None:
        should_show = (output is None)
    else:
        should_show = show
    
    # Save figure if output path specified
    if output is not None:
        print(f"\nSaving figure to: {output}")
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print("Figure saved!")
    
    # Show plot
    if should_show:
        print("\nShowing interactive plot...")
        plt.show()
    else:
        plt.close()
        print("\nPlot saved. Use --show to display interactive plot.")

if __name__ == '__main__':
    main()

