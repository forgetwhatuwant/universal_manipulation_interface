#!/usr/bin/env python3
"""
Find recommended initial pose for Piper robot based on trajectory analysis.

This script analyzes the camera trajectory and suggests safe initial poses
for replaying on Piper robot.

Usage:
    python scripts_real/find_recommended_piper_pose.py \
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

import click
import numpy as np
import pandas as pd
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.piper_interpolation_controller import PiperInterpolationController

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
    
    return poses, timestamps, df

def get_piper_fk_init_pose():
    """
    Get Piper forward kinematics default initial pose.
    From piper_fk.py: [56.128, 0.0, 213.266, 0.0, 85.0, 0.0] in mm/degrees
    Converted to meters/radians.
    """
    # Position in mm -> meters
    pos_mm = np.array([56.128, 0.0, 213.266])
    pos_m = pos_mm / 1000.0
    
    # Rotation in degrees -> radians -> rotation vector
    # RPY: [0.0, 85.0, 0.0] degrees
    rpy_deg = np.array([0.0, 85.0, 0.0])
    rpy_rad = np.radians(rpy_deg)
    rot = st.Rotation.from_euler('XYZ', rpy_rad)
    rot_vec = rot.as_rotvec()
    
    pose = np.concatenate([pos_m, rot_vec])
    return pose

@click.command()
@click.option('--csv_path', '-i', required=True, type=str,
              help='Path to camera trajectory CSV file')
@click.option('--can_name', default='can0',
              help='CAN interface name (e.g., can0)')
@click.option('--get_current', is_flag=True, default=False,
              help='Get current robot pose for comparison')
def main(csv_path, can_name, get_current):
    """
    Find recommended initial pose for Piper robot.
    """
    print("=" * 80)
    print("Piper Initial Pose Recommendation")
    print("=" * 80)
    
    # Load trajectory
    poses, timestamps, df = load_camera_trajectory(csv_path)
    
    # Analyze trajectory
    positions = poses[:, :3]
    rotations = poses[:, 3:]
    
    first_pose = poses[0]
    center_pose = np.mean(poses, axis=0)
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    mean_pos = np.mean(positions, axis=0)
    
    # Get Piper FK default pose
    fk_init_pose = get_piper_fk_init_pose()
    
    print("\n" + "=" * 80)
    print("Trajectory Analysis")
    print("=" * 80)
    print(f"Total poses: {len(poses)}")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    print(f"\nPosition bounds:")
    print(f"  X: [{min_pos[0]:.4f}, {max_pos[0]:.4f}] m ({min_pos[0]*1000:.1f} to {max_pos[0]*1000:.1f} mm)")
    print(f"  Y: [{min_pos[1]:.4f}, {max_pos[1]:.4f}] m ({min_pos[1]*1000:.1f} to {max_pos[1]*1000:.1f} mm)")
    print(f"  Z: [{min_pos[2]:.4f}, {max_pos[2]:.4f}] m ({min_pos[2]*1000:.1f} to {max_pos[2]*1000:.1f} mm)")
    print(f"\nTrajectory center:")
    print(f"  Position: [{mean_pos[0]:.4f}, {mean_pos[1]:.4f}, {mean_pos[2]:.4f}] m")
    
    print("\n" + "=" * 80)
    print("Recommended Initial Poses")
    print("=" * 80)
    
    # Option 1: Trajectory's first pose
    print("\n1. Trajectory's First Pose (exact match):")
    print(f"   Position: [{first_pose[0]:.4f}, {first_pose[1]:.4f}, {first_pose[2]:.4f}] m")
    print(f"   Rotation: [{first_pose[3]:.4f}, {first_pose[4]:.4f}, {first_pose[5]:.4f}] rad")
    print(f"   Distance from origin: {np.linalg.norm(first_pose[:3]):.4f} m")
    print(f"   ✓ Best for: Exact trajectory replay")
    print(f"   ⚠ Note: May require large movement from current pose")
    
    # Option 2: Piper FK default pose
    print("\n2. Piper Forward Kinematics Default Pose:")
    print(f"   Position: [{fk_init_pose[0]:.4f}, {fk_init_pose[1]:.4f}, {fk_init_pose[2]:.4f}] m")
    print(f"   Rotation: [{fk_init_pose[3]:.4f}, {fk_init_pose[4]:.4f}, {fk_init_pose[5]:.4f}] rad")
    print(f"   Distance from origin: {np.linalg.norm(fk_init_pose[:3]):.4f} m")
    print(f"   ✓ Best for: Safe home position, standard starting point")
    print(f"   ⚠ Note: May need to adjust trajectory to be relative to this pose")
    
    # Option 3: Trajectory center
    print("\n3. Trajectory Center (average):")
    print(f"   Position: [{center_pose[0]:.4f}, {center_pose[1]:.4f}, {center_pose[2]:.4f}] m")
    print(f"   Rotation: [{center_pose[3]:.4f}, {center_pose[4]:.4f}, {center_pose[5]:.4f}] rad")
    print(f"   Distance from origin: {np.linalg.norm(center_pose[:3]):.4f} m")
    print(f"   ✓ Best for: Balanced starting point")
    
    # Option 4: Safe pose near trajectory
    safe_z = max(min_pos[2] + 0.05, 0.15)  # At least 5cm above min, or 15cm minimum
    safe_pose = np.array([
        mean_pos[0],  # Use mean X
        mean_pos[1],  # Use mean Y
        safe_z,       # Safe Z height
        first_pose[3], first_pose[4], first_pose[5]  # Use first rotation
    ])
    
    print("\n4. Safe Pose (near trajectory, elevated):")
    print(f"   Position: [{safe_pose[0]:.4f}, {safe_pose[1]:.4f}, {safe_pose[2]:.4f}] m")
    print(f"   Rotation: [{safe_pose[3]:.4f}, {safe_pose[4]:.4f}, {safe_pose[5]:.4f}] rad")
    print(f"   Distance from origin: {np.linalg.norm(safe_pose[:3]):.4f} m")
    print(f"   ✓ Best for: Safe starting point above workspace")
    
    # Get current robot pose if requested
    if get_current:
        print("\n" + "=" * 80)
        print("Current Robot Pose")
        print("=" * 80)
        try:
            with SharedMemoryManager() as shm_manager:
                with PiperInterpolationController(
                        shm_manager=shm_manager,
                        can_name=can_name,
                        frequency=100,
                        max_pos_speed=0.25,
                        max_rot_speed=0.6,
                        verbose=False) as controller:
                    # Controller is already ready (start() waits in context manager)
                    import time
                    time.sleep(1.0)  # Give it a moment to stabilize
                    current_state = controller.get_state()
                    current_pose = current_state['ee_pose']
                    
                    print(f"\nCurrent robot pose:")
                    print(f"  Position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}] m")
                    print(f"  Rotation: [{current_pose[3]:.4f}, {current_pose[4]:.4f}, {current_pose[5]:.4f}] rad")
                    print(f"  Distance from origin: {np.linalg.norm(current_pose[:3]):.4f} m")
                    
                    # Calculate distances to recommended poses
                    print(f"\nDistances from current pose:")
                    dist_to_first = np.linalg.norm(current_pose[:3] - first_pose[:3])
                    dist_to_fk = np.linalg.norm(current_pose[:3] - fk_init_pose[:3])
                    dist_to_center = np.linalg.norm(current_pose[:3] - center_pose[:3])
                    dist_to_safe = np.linalg.norm(current_pose[:3] - safe_pose[:3])
                    
                    print(f"  To trajectory first: {dist_to_first:.4f} m ({dist_to_first*1000:.1f} mm)")
                    print(f"  To FK default: {dist_to_fk:.4f} m ({dist_to_fk*1000:.1f} mm)")
                    print(f"  To trajectory center: {dist_to_center:.4f} m ({dist_to_center*1000:.1f} mm)")
                    print(f"  To safe pose: {dist_to_safe:.4f} m ({dist_to_safe*1000:.1f} mm)")
                    
                    # Recommend based on distance
                    print(f"\n💡 Recommendation:")
                    if dist_to_first < 0.1:
                        print(f"   Use trajectory's first pose (close to current position)")
                    elif dist_to_safe < 0.15:
                        print(f"   Use safe pose (close and safe)")
                    elif dist_to_fk < 0.2:
                        print(f"   Use FK default pose (close to standard home)")
                    else:
                        print(f"   Consider using --use_current_pose flag to start from current position")
                        print(f"   Or move robot closer to trajectory first pose ({dist_to_first*1000:.1f} mm away)")
        except Exception as e:
            print(f"⚠ Could not get current robot pose: {e}")
            print("   Make sure robot is connected and CAN interface is correct")
    
    print("\n" + "=" * 80)
    print("Usage Recommendations")
    print("=" * 80)
    print("\nFor SAFEST and SLOWEST replay:")
    print("  1. Use --use_current_pose flag to start from current position")
    print("  2. Use --slow_down 3.0 or higher for very slow movement")
    print("  3. Use --max_pos_speed 0.10 for very slow position movement")
    print("  4. Use --max_rot_speed 0.3 for very slow rotation")
    print("  5. Use --check_bounds to verify trajectory is within workspace")
    print("\nExample commands:")
    print("\n  SLOWEST (safest for first test):")
    print(f"  python scripts_real/replay_piper_camera_trajectory.py \\")
    print(f"    --csv_path {csv_path} \\")
    print(f"    --can_name {can_name} \\")
    print(f"    --slow_down 3.0 \\")
    print(f"    --max_pos_speed 0.10 \\")
    print(f"    --max_rot_speed 0.3 \\")
    print(f"    --use_current_pose")
    print("\n  MODERATE (default, slow and safe):")
    print(f"  python scripts_real/replay_piper_camera_trajectory.py \\")
    print(f"    --csv_path {csv_path} \\")
    print(f"    --can_name {can_name} \\")
    print(f"    --slow_down 2.0 \\")
    print(f"    --use_current_pose")
    print("\n  Note: Defaults are now set for slow, safe movement!")
    print("=" * 80)

if __name__ == '__main__':
    main()

