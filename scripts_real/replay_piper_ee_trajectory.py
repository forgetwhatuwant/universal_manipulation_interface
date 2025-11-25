#!/usr/bin/env python3
"""
Replay Piper EE trajectory CSV on Piper robot using UMI SharedMemory system.

This script reads a Piper EE trajectory CSV file (with rotation vector poses) and replays
it on the Piper robot using UMI's SharedMemory-based control system.

CSV Format Expected:
    timestamp,x,y,z,rx,ry,rz

Usage:
    python scripts_real/replay_piper_ee_trajectory.py \
        --csv_path piper_ee_trajectory.csv \
        --can_name can0 \
        --output replay_result.pkl \
        --save_initial_joints initial_joints.npy
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

import time
import click
import numpy as np
import pandas as pd
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.piper_interpolation_controller import PiperInterpolationController
from umi.common.precise_sleep import precise_sleep, precise_wait

def quaternion_to_rotvec(qx, qy, qz, qw):
    """Convert quaternion [x, y, z, w] to rotation vector [rx, ry, rz]."""
    quat = np.array([qx, qy, qz, qw])
    rot = st.Rotation.from_quat(quat)
    return rot.as_rotvec()

def load_trajectory(csv_path):
    """
    Load trajectory from CSV file. Auto-detects format:
    - Camera trajectory: has 'q_x', 'q_y', 'q_z', 'q_w' columns
    - EE trajectory: has 'rx', 'ry', 'rz' columns
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        tuple: (poses, timestamps, is_camera_traj) where poses is (N, 6) array [x,y,z,rx,ry,rz]
               and timestamps is (N,) array in seconds
    """
    print(f"Loading trajectory from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check if it's a camera trajectory (has quaternions)
    if 'q_x' in df.columns and 'q_y' in df.columns and 'q_z' in df.columns and 'q_w' in df.columns:
        print("Detected CAMERA trajectory format (quaternions)")
        positions = df[['x', 'y', 'z']].values
        quaternions = df[['q_x', 'q_y', 'q_z', 'q_w']].values
        timestamps = df['timestamp'].values
        
        # Convert quaternions to rotation vectors
        rotation_vectors = np.array([
            quaternion_to_rotvec(q[0], q[1], q[2], q[3])
            for q in quaternions
        ])
        
        poses = np.hstack([positions, rotation_vectors])
        is_camera_traj = True
    elif 'rx' in df.columns and 'ry' in df.columns and 'rz' in df.columns:
        print("Detected EE trajectory format (rotation vectors)")
        poses = df[['x', 'y', 'z', 'rx', 'ry', 'rz']].values
        timestamps = df['timestamp'].values
        is_camera_traj = False
    else:
        raise ValueError("Unknown trajectory format. Expected either camera trajectory (q_x, q_y, q_z, q_w) or EE trajectory (rx, ry, rz)")
    
    print(f"Loaded {len(poses)} poses")
    print(f"  Time range: {timestamps[0]:.3f} - {timestamps[-1]:.3f} seconds")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    
    return poses, timestamps, is_camera_traj

@click.command()
@click.option('--csv_path', '-i', required=True, type=str, 
              help='Path to EE trajectory CSV file')
@click.option('--can_name', default='can0', 
              help='CAN interface name (e.g., can0)')
@click.option('--frequency', '-f', default=100, type=float,
              help='Control frequency in Hz')
@click.option('--slow_down', default=2.0, type=float,
              help='Slow down factor (1.0 = real-time, 2.0 = half speed). Default: 2.0 for slow, safe movement')
@click.option('--control_frequency', default=50.0, type=float,
              help='Control command frequency (Hz). Default: 50.0 Hz. Higher = smoother, but needs faster comms')
@click.option('--command_lookahead', default=0.1, type=float,
              help='Command lookahead time in seconds')
@click.option('--max_pos_speed', default=0.25, type=float,
              help='Maximum position speed (m/s). Default: 0.25 m/s')
@click.option('--max_rot_speed', default=0.6, type=float,
              help='Maximum rotation speed (rad/s). Default: 0.6 rad/s')
@click.option('--output', '-o', type=str, default=None,
              help='Optional: Save replay results to pickle file')
@click.option('--save_initial_joints', type=str, default=None,
              help='Save initial joint positions to numpy file')
@click.option('--check_bounds', is_flag=True, default=True,
              help='Check trajectory bounds before execution')
@click.option('--max_reach', default=0.5, type=float,
              help='Maximum reach distance from robot base (m) for safety check')
@click.option('--use_incremental', is_flag=True, default=False,
              help='Use incremental/delta-based control (start from current robot pose)')
def main(csv_path, can_name, frequency, slow_down, control_frequency,
         command_lookahead, max_pos_speed, max_rot_speed, output,
         save_initial_joints, check_bounds, max_reach, use_incremental):
    """
    Replay trajectory on Piper robot. Supports both camera trajectory (quaternions) 
    and EE trajectory (rotation vectors) formats. Can use incremental/delta-based 
    control for camera trajectories with coordinate frame transformation.
    """
    print("=" * 80)
    print("Piper Trajectory Replay")
    print("=" * 80)
    
    # Load trajectory (auto-detects format)
    poses, timestamps, is_camera_traj = load_trajectory(csv_path)
    
    # Safety: Check trajectory bounds
    if check_bounds:
        positions = poses[:, :3]
        distances = np.linalg.norm(positions, axis=1)
        max_distance = np.max(distances)
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        
        print("\n" + "=" * 80)
        print("Trajectory Safety Check")
        print("=" * 80)
        print(f"Position bounds:")
        print(f"  X: [{min_pos[0]:.4f}, {max_pos[0]:.4f}] m")
        print(f"  Y: [{min_pos[1]:.4f}, {max_pos[1]:.4f}] m")
        print(f"  Z: [{min_pos[2]:.4f}, {max_pos[2]:.4f}] m")
        print(f"  Max distance from origin: {max_distance:.4f} m")
        print(f"  Safety limit: {max_reach} m")
        
        if max_distance > max_reach:
            print(f"\n⚠️  WARNING: Trajectory exceeds safety limit!")
            print(f"   Maximum distance ({max_distance:.4f} m) > limit ({max_reach} m)")
            if not click.confirm('   Continue anyway?', default=False):
                print("Aborted for safety.")
                return
        else:
            print("✓ Trajectory within safety bounds")
        print("=" * 80)
    
    # Apply slow down factor (warn if too fast)
    if slow_down < 1.5:
        print("\n⚠️  WARNING: Slow down factor is less than 1.5x")
        print(f"   Current: {slow_down}x (faster than recommended)")
        print("   Recommended: --slow_down 2.0 or higher for safe, slow movement")
        if not click.confirm('   Continue with fast speed?', default=False):
            print("Aborted. Use --slow_down 2.0 or higher for safer, slower movement")
            return
    
    if max_pos_speed > 0.2:
        print("\n⚠️  WARNING: Maximum position speed is high")
        print(f"   Current: {max_pos_speed} m/s")
        print("   Recommended: --max_pos_speed 0.15 or lower for slow, safe movement")
        if not click.confirm('   Continue with high speed?', default=False):
            print("Aborted. Use --max_pos_speed 0.15 for slower, safer movement")
            return
    
    # Apply slow_down to timestamps
    timestamps = timestamps * slow_down
    
    # Calculate trajectory properties
    duration_sec = timestamps[-1] - timestamps[0]
    dt = np.mean(np.diff(timestamps))
    avg_fps = 1.0 / dt
    
    print(f"\nReplay Configuration:")
    print(f"  Trajectory type: {'Camera trajectory' if is_camera_traj else 'EE trajectory'}")
    print(f"  Control mode: {'Incremental (delta-based)' if (use_incremental or is_camera_traj) else 'Absolute poses'}")
    print(f"  CAN interface: {can_name}")
    print(f"  Control frequency: {frequency} Hz")
    print(f"  Slow down factor: {slow_down}x {'(SLOW & SAFE)' if slow_down >= 2.0 else '(FAST)'}")
    print(f"  Trajectory duration: {duration_sec:.3f} seconds")
    print(f"  Average trajectory FPS: {avg_fps:.2f}")
    print(f"  Command lookahead: {command_lookahead:.3f} seconds")
    print(f"  Max position speed: {max_pos_speed} m/s")
    print(f"  Max rotation speed: {max_rot_speed} rad/s ({np.degrees(max_rot_speed):.1f} deg/s)")
    print(f"  Control frequency: {control_frequency} Hz")
    print("=" * 80)
    
    # Wait for user confirmation
    click.confirm('\nReady to start replay?', abort=True)
    
    # Setup SharedMemoryManager and controller
    with SharedMemoryManager() as shm_manager:
        with PiperInterpolationController(
                shm_manager=shm_manager,
                can_name=can_name,
                frequency=frequency,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                verbose=True) as controller:
            
            # Controller is already ready (start() waits in context manager)
            # But give it a moment to stabilize
            print("\nController initialized, waiting for stabilization...")
            time.sleep(1.0)
            print("Ready!")
            
            # Get current robot pose and joint positions
            initial_state = controller.get_state()
            current_pose = initial_state['ActualTCPPose']
            initial_joints = initial_state['ActualQ']
            
            print(f"\nCurrent robot pose:")
            print(f"  Position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}] m")
            print(f"  Rotation: [{current_pose[3]:.4f}, {current_pose[4]:.4f}, {current_pose[5]:.4f}] rad")
            print(f"\nCurrent joint positions:")
            print(f"  Joints: {initial_joints}")
            print(f"  Joints (deg): {np.degrees(initial_joints)}")
            
            # Save initial joints if requested
            if save_initial_joints:
                np.save(save_initial_joints, initial_joints)
                print(f"\n✓ Saved initial joint positions to: {save_initial_joints}")
            
            # Handle camera trajectory with coordinate frame transformation and incremental control
            r_align = None  # Initialize for camera trajectory alignment
            r_traj_first = None  # Initialize for rotation delta calculation
            
            if is_camera_traj or use_incremental:
                # Use incremental control: calculate deltas relative to first frame
                print(f"\nUsing INCREMENTAL CONTROL (delta-based)")
                if is_camera_traj:
                    print(f"  Camera trajectory detected - applying coordinate frame transformation")
                print(f"  First trajectory frame will be treated as reference (delta = 0)")
                print(f"  Robot will start from current pose and follow relative motion")
                
                # Save robot's initial pose as reference
                robot_initial_pose = current_pose.copy()
                print(f"\nRobot initial pose (reference):")
                print(f"  Position: [{robot_initial_pose[0]:.4f}, {robot_initial_pose[1]:.4f}, {robot_initial_pose[2]:.4f}] m")
                print(f"  Rotation: [{robot_initial_pose[3]:.4f}, {robot_initial_pose[4]:.4f}, {robot_initial_pose[5]:.4f}] rad")
                
                # Transform camera frame to robot base frame if needed
                if is_camera_traj:
                    print(f"\nTransforming camera trajectory to robot base frame...")
                    print(f"  Camera frame: Z-forward, X-right, Y-down")
                    print(f"  Robot frame: X-forward, Y-left/right, Z-up")
                    
                    camera_to_robot_transform = np.array([
                        [0, 0, 1],   # Camera Z (forward) → Robot X (forward)
                        [1, 0, 0],   # Camera X (right) → Robot Y (sideways)
                        [0, -1, 0],  # Camera Y (down) → Robot Z (up)
                    ])
                    
                    # Transform positions
                    for i in range(len(poses)):
                        poses[i, :3] = camera_to_robot_transform @ poses[i, :3]
                    
                    # Transform rotations
                    for i in range(len(poses)):
                        if np.linalg.norm(poses[i, 3:]) > 1e-6:
                            r_cam = st.Rotation.from_rotvec(poses[i, 3:])
                            r_mat_cam = r_cam.as_matrix()
                            r_mat_robot = camera_to_robot_transform @ r_mat_cam @ camera_to_robot_transform.T
                            r_robot = st.Rotation.from_matrix(r_mat_robot)
                            poses[i, 3:] = r_robot.as_rotvec()
                    
                    # Align initial rotation
                    r_cam_first = st.Rotation.from_rotvec(poses[0, 3:])
                    r_robot_start = st.Rotation.from_rotvec(robot_initial_pose[3:])
                    r_align = r_robot_start * r_cam_first.inv()
                    print(f"  Alignment rotation: {r_align.as_rotvec()}")
                
                # Calculate deltas relative to first frame
                print(f"\nCalculating trajectory deltas relative to first frame...")
                deltas = np.zeros_like(poses)
                deltas[0] = np.zeros(6)  # First frame: no movement
                
                r_traj_first = st.Rotation.from_rotvec(poses[0, 3:])
                
                for i in range(1, len(poses)):
                    # Position delta
                    deltas[i, :3] = poses[i, :3] - poses[0, :3]
                    
                    # Rotation delta
                    r_traj_i = st.Rotation.from_rotvec(poses[i, 3:])
                    r_delta = r_traj_i * r_traj_first.inv()
                    deltas[i, 3:] = r_delta.as_rotvec()
                
                max_pos_delta = np.max(np.linalg.norm(deltas[:, :3], axis=1))
                max_rot_delta = np.max(np.linalg.norm(deltas[:, 3:], axis=1))
                print(f"  Max position delta: {max_pos_delta:.4f} m")
                print(f"  Max rotation delta: {np.degrees(max_rot_delta):.2f} deg")
                
                if max_pos_delta > 0.5:
                    print(f"\n⚠️  WARNING: Large position deltas detected ({max_pos_delta:.4f} m)")
                    if not click.confirm('   Continue anyway?', default=False):
                        print("Aborted.")
                        return
            else:
                # Absolute pose control (for pre-generated EE trajectories)
                first_pose = poses[0]
                pos_diff = np.linalg.norm(first_pose[:3] - current_pose[:3])
                print(f"\nDistance from current pose to trajectory start: {pos_diff:.4f} m ({pos_diff*1000:.1f} mm)")
                
                if pos_diff > 0.2:
                    print(f"\n⚠️  WARNING: Large movement required ({pos_diff:.4f} m)")
                    if not click.confirm('   Continue anyway?', default=False):
                        print("Aborted.")
                        return
            
            # Start replay
            print("\n" + "=" * 80)
            print("Starting trajectory replay...")
            print("=" * 80)
            
            t_start = time.time()
            target_pose_traj = []
            target_timestamps = []
            actual_pose_traj = []
            actual_q_traj = []
            actual_qd_traj = []
            actual_timestamps = []
            
            # Calculate cycle time for control loop
            control_dt = 1.0 / control_frequency
            print(f"\nSending commands at {control_frequency} Hz (every {control_dt:.3f}s)")
            
            # Apply slow_down to trajectory timestamps
            raw_duration = timestamps[-1] - timestamps[0]
            total_duration = raw_duration * slow_down
            
            # Scale timestamps by slow_down factor (keep relative to first timestamp)
            scaled_timestamps = timestamps[0] + (timestamps - timestamps[0]) * slow_down
            
            # Interpolate trajectory to match control frequency
            # Create target_times as relative times starting from 0
            num_steps = max(int(total_duration * control_frequency), len(poses))
            target_times = np.linspace(0.0, total_duration, num_steps)
            
            from scipy.interpolate import interp1d
            
            if is_camera_traj or use_incremental:
                # Interpolate DELTAS (incremental control) using scaled timestamps
                # Convert target_times (relative) to absolute timestamps for interpolation
                target_times_absolute = timestamps[0] + target_times
                
                pos_delta_interp = interp1d(scaled_timestamps, deltas[:, :3], axis=0, kind='linear', 
                                           fill_value='extrapolate', assume_sorted=True)
                rot_delta_interp = interp1d(scaled_timestamps, deltas[:, 3:], axis=0, kind='linear',
                                           fill_value='extrapolate', assume_sorted=True)
                
                interpolated_deltas = np.hstack([
                    pos_delta_interp(target_times_absolute),
                    rot_delta_interp(target_times_absolute)
                ])
                
                print(f"Interpolated {len(interpolated_deltas)} delta commands from {len(deltas)} trajectory points")
                print(f"Raw trajectory duration: {raw_duration:.3f} seconds")
                print(f"Scaled replay duration: {total_duration:.3f} seconds (slow_down: {slow_down}x)\n")
                
                # Reset start time
                t_start = time.time()
                
                for iter_idx in range(len(interpolated_deltas)):
                    # Calculate cycle end time
                    t_cycle_end = t_start + (iter_idx + 1) * control_dt
                    
                    # Calculate target time for this waypoint (relative to t_start)
                    # target_times is now relative (0 to total_duration)
                    rel_time = target_times[iter_idx]
                    this_t_target = t_start + rel_time + command_lookahead
                    
                    # Get interpolated delta
                    delta = interpolated_deltas[iter_idx]
                    
                    # Calculate target pose: robot_initial_pose + delta
                    target_pos = robot_initial_pose[:3] + delta[:3]
                    
                    # Rotation: compose rotations
                    r_initial = st.Rotation.from_rotvec(robot_initial_pose[3:])
                    r_delta = st.Rotation.from_rotvec(delta[3:])
                    
                    if is_camera_traj and r_align is not None:
                        # For camera trajectories: apply delta relative to transformed first frame, then align
                        # r_delta is relative to first frame (already transformed to robot frame)
                        # r_after_delta = r_delta * r_traj_first gives absolute rotation at that point
                        # r_target = r_align * r_after_delta aligns to robot's initial orientation
                        r_after_delta = r_delta * r_traj_first
                        r_target = r_align * r_after_delta
                    else:
                        # For EE trajectories or incremental without camera: apply delta to robot initial pose
                        r_target = r_delta * r_initial
                    
                    target_rot = r_target.as_rotvec()
                    this_pose = np.concatenate([target_pos, target_rot])
                    
                    # Schedule waypoint
                    controller.schedule_waypoint(
                        pose=this_pose,
                        target_time=this_t_target
                    )
                    
                    # Record target trajectory
                    target_pose_traj.append(this_pose.copy())
                    target_timestamps.append(this_t_target)
                    
                    # Get actual pose (non-blocking)
                    try:
                        state = controller.get_state()
                        actual_pose_traj.append(state['ActualTCPPose'].copy())
                        actual_q_traj.append(state['ActualQ'].copy())
                        actual_qd_traj.append(state['ActualQd'].copy())
                        actual_timestamps.append(time.time())
                    except:
                        pass
                    
                    # Wait until cycle end
                    precise_wait(t_cycle_end, time_func=time.time)
                    
                    # Progress indicator (every 1 second)
                    if (iter_idx + 1) % int(control_frequency) == 0:
                        progress = (iter_idx + 1) / len(interpolated_deltas) * 100
                        elapsed = time.time() - t_start
                        print(f"Progress: {progress:.1f}% ({iter_idx + 1}/{len(interpolated_deltas)}) | "
                              f"Elapsed: {elapsed:.2f}s | "
                              f"Remaining: ~{total_duration - elapsed:.2f}s")
            else:
                # Absolute pose control (for EE trajectories) using scaled timestamps
                # Convert target_times (relative) to absolute timestamps for interpolation
                target_times_absolute = timestamps[0] + target_times
                
                pos_interp = interp1d(scaled_timestamps, poses[:, :3], axis=0, kind='linear', 
                                     fill_value='extrapolate', assume_sorted=True)
                rot_interp = interp1d(scaled_timestamps, poses[:, 3:], axis=0, kind='linear',
                                     fill_value='extrapolate', assume_sorted=True)
                
                interpolated_poses = np.hstack([
                    pos_interp(target_times_absolute),
                    rot_interp(target_times_absolute)
                ])
                
                print(f"Interpolated {len(interpolated_poses)} commands from {len(poses)} trajectory points")
                print(f"Raw trajectory duration: {raw_duration:.3f} seconds")
                print(f"Scaled replay duration: {total_duration:.3f} seconds (slow_down: {slow_down}x)\n")
                
                # Reset start time
                t_start = time.time()
                
                for iter_idx in range(len(interpolated_poses)):
                    # Calculate cycle end time
                    t_cycle_end = t_start + (iter_idx + 1) * control_dt
                    
                    # Calculate target time for this waypoint (relative to t_start)
                    # target_times is now relative (0 to total_duration)
                    rel_time = target_times[iter_idx]
                    this_t_target = t_start + rel_time + command_lookahead
                    
                    # Get interpolated pose for this step
                    this_pose = interpolated_poses[iter_idx]
                    
                    # Schedule waypoint
                    controller.schedule_waypoint(
                        pose=this_pose,
                        target_time=this_t_target
                    )
                    
                    # Record target trajectory
                    target_pose_traj.append(this_pose.copy())
                    target_timestamps.append(this_t_target)
                    
                    # Get actual pose (non-blocking)
                    try:
                        state = controller.get_state()
                        actual_pose_traj.append(state['ActualTCPPose'].copy())
                        actual_q_traj.append(state['ActualQ'].copy())
                        actual_qd_traj.append(state['ActualQd'].copy())
                        actual_timestamps.append(time.time())
                    except:
                        pass
                    
                    # Wait until cycle end
                    precise_wait(t_cycle_end, time_func=time.time)
                    
                    # Progress indicator (every 1 second)
                    if (iter_idx + 1) % int(control_frequency) == 0:
                        progress = (iter_idx + 1) / len(interpolated_poses) * 100
                        elapsed = time.time() - t_start
                        print(f"Progress: {progress:.1f}% ({iter_idx + 1}/{len(interpolated_poses)}) | "
                              f"Elapsed: {elapsed:.2f}s | "
                              f"Remaining: ~{total_duration - elapsed:.2f}s")
            
            print("\n" + "=" * 80)
            print("Trajectory replay completed!")
            print("=" * 80)
            
            # Get final state
            final_state = controller.get_all_state()
            
            # Save results if requested
            if output is not None:
                import pickle
                result = {
                    'target_poses': np.array(target_pose_traj),
                    'target_timestamps': np.array(target_timestamps),
                    'actual_poses': np.array(actual_pose_traj) if actual_pose_traj else None,
                    'actual_q': np.array(actual_q_traj) if actual_q_traj else None,
                    'actual_qd': np.array(actual_qd_traj) if actual_qd_traj else None,
                    'actual_timestamps': np.array(actual_timestamps) if actual_timestamps else None,
                    'initial_joints': initial_joints,
                    'final_joints': final_state.get('ActualQ'),
                    'robot_state': final_state,
                    'config': {
                        'csv_path': csv_path,
                        'can_name': can_name,
                        'frequency': frequency,
                        'slow_down': slow_down,
                        'max_pos_speed': max_pos_speed,
                        'max_rot_speed': max_rot_speed,
                    }
                }
                print(f"\nSaving results to: {output}")
                pickle.dump(result, open(output, 'wb'))
                print("Results saved!")
            
            print("\nReplay finished. Robot will hold final pose.")
            print("Press Ctrl+C to stop.")

if __name__ == '__main__':
    main()

