#!/usr/bin/env python3
"""
Replay camera trajectory CSV to Piper robot using UMI SharedMemory system.

This script reads a camera trajectory CSV file (with quaternion poses) and replays
it on the Piper robot using UMI's SharedMemory-based control system.

CSV Format Expected:
    frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w

Usage:
    python scripts_real/replay_piper_camera_trajectory.py \
        --csv_path example_demo_session/demos/demo_*/camera_trajectory.csv \
        --can_name can0 \
        --frequency 100 \
        --slow_down 1.0
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

import time
import click
import numpy as np
import pandas as pd
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.piper_interpolation_controller import PiperInterpolationController
from umi.common.precise_sleep import precise_sleep, precise_wait

def quaternion_to_rotvec(qx, qy, qz, qw):
    """
    Convert quaternion [x, y, z, w] to rotation vector [rx, ry, rz].
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        np.array([rx, ry, rz]): Rotation vector in radians
    """
    # scipy expects [x, y, z, w] format
    quat = np.array([qx, qy, qz, qw])
    rot = st.Rotation.from_quat(quat)
    return rot.as_rotvec()

def load_camera_trajectory(csv_path):
    """
    Load camera trajectory from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        tuple: (poses, timestamps) where poses is (N, 6) array [x,y,z,rx,ry,rz]
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
    
    print(f"Loaded {len(poses)} poses")
    print(f"  Time range: {timestamps[0]:.3f} - {timestamps[-1]:.3f} seconds")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    print(f"  Average FPS: {1.0 / np.mean(np.diff(timestamps)):.2f}")
    
    return poses, timestamps

@click.command()
@click.option('--csv_path', '-i', required=True, type=str, 
              help='Path to camera trajectory CSV file')
@click.option('--can_name', default='can0', 
              help='CAN interface name (e.g., can0)')
@click.option('--frequency', '-f', default=100, type=float,
              help='Control frequency in Hz')
@click.option('--slow_down', default=2.0, type=float,
              help='Slow down factor (1.0 = real-time, 2.0 = half speed). Default: 2.0 for slow, safe movement')
@click.option('--control_frequency', default=50.0, type=float,
              help='Control command frequency (Hz). Default: 50.0 Hz. Higher = smoother, but needs faster comms')
@click.option('--init_pose_sec', default=5.0, type=float,
              help='Time to move to initial pose before starting replay (default: 5.0s)')
@click.option('--command_lookahead', default=0.1, type=float,
              help='Command lookahead time in seconds')
@click.option('--max_pos_speed', default=0.25, type=float,
              help='Maximum position speed (m/s). Default: 0.25 m/s')
@click.option('--max_rot_speed', default=0.6, type=float,
              help='Maximum rotation speed (rad/s). Default: 0.6 rad/s')
@click.option('--output', '-o', type=str, default=None,
              help='Optional: Save replay results to pickle file')
@click.option('--use_current_pose', is_flag=True, default=False,
              help='Use current robot pose as starting point instead of trajectory first pose')
@click.option('--check_bounds', is_flag=True, default=True,
              help='Check trajectory bounds before execution')
@click.option('--max_reach', default=0.5, type=float,
              help='Maximum reach distance from robot base (m) for safety check')
def main(csv_path, can_name, frequency, slow_down, control_frequency, init_pose_sec,
         command_lookahead, max_pos_speed, max_rot_speed, output,
         use_current_pose, check_bounds, max_reach):
    """
    Replay camera trajectory on Piper robot.
    """
    print("=" * 80)
    print("Piper Camera Trajectory Replay")
    print("=" * 80)
    
    # Load trajectory
    poses, timestamps = load_camera_trajectory(csv_path)
    
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
    
    timestamps = timestamps * slow_down
    
    # Calculate trajectory properties
    duration_sec = timestamps[-1] - timestamps[0]
    dt = np.mean(np.diff(timestamps))
    avg_fps = 1.0 / dt
    
    print(f"\nReplay Configuration:")
    print(f"  CAN interface: {can_name}")
    print(f"  Control frequency: {frequency} Hz")
    print(f"  Slow down factor: {slow_down}x {'(SLOW & SAFE)' if slow_down >= 2.0 else '(FAST)'}")
    print(f"  Trajectory duration: {duration_sec:.3f} seconds")
    print(f"  Average trajectory FPS: {avg_fps:.2f}")
    print(f"  Command lookahead: {command_lookahead:.3f} seconds")
    print(f"  Max position speed: {max_pos_speed} m/s")
    print(f"  Max rotation speed: {max_rot_speed} rad/s ({np.degrees(max_rot_speed):.1f} deg/s)")
    print(f"  Control frequency: {control_frequency} Hz")
    print(f"  Use current pose as start: {use_current_pose}")
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
            
            # Get current robot pose
            initial_state = controller.get_state()
            current_pose = initial_state['ActualTCPPose']
            print(f"\nCurrent robot pose:")
            print(f"  Position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}] m")
            print(f"  Rotation: [{current_pose[3]:.4f}, {current_pose[4]:.4f}, {current_pose[5]:.4f}] rad")
            
            # Use incremental control: calculate deltas relative to first frame
            print(f"\nUsing INCREMENTAL CONTROL (delta-based)")
            print(f"  First trajectory frame will be treated as reference (delta = 0)")
            print(f"  Robot will start from current pose and follow relative motion")
            
            # Save robot's initial pose as reference
            robot_initial_pose = current_pose.copy()
            print(f"\nRobot initial pose (reference):")
            print(f"  Position: [{robot_initial_pose[0]:.4f}, {robot_initial_pose[1]:.4f}, {robot_initial_pose[2]:.4f}] m")
            print(f"  Rotation: [{robot_initial_pose[3]:.4f}, {robot_initial_pose[4]:.4f}, {robot_initial_pose[5]:.4f}] rad")
            
            # Calculate deltas relative to first frame
            print(f"\nCalculating trajectory deltas relative to first frame...")
            deltas = np.zeros_like(poses)
            deltas[0] = np.zeros(6)  # First frame: no movement (delta = 0)
            
            # Get first frame rotation for reference
            r_traj_first = st.Rotation.from_rotvec(poses[0, 3:])
            
            for i in range(1, len(poses)):
                # Position delta: simple subtraction
                deltas[i, :3] = poses[i, :3] - poses[0, :3]
                
                # Rotation delta: relative rotation from first frame
                r_traj_i = st.Rotation.from_rotvec(poses[i, 3:])
                r_delta = r_traj_i * r_traj_first.inv()  # Relative rotation: R_i * R_0^-1
                deltas[i, 3:] = r_delta.as_rotvec()
            
            # Calculate trajectory statistics
            max_pos_delta = np.max(np.linalg.norm(deltas[:, :3], axis=1))
            max_rot_delta = np.max(np.linalg.norm(deltas[:, 3:], axis=1))
            print(f"  Max position delta: {max_pos_delta:.4f} m")
            print(f"  Max rotation delta: {np.degrees(max_rot_delta):.2f} deg")
            print(f"  Total frames: {len(poses)}")
            
            # Check if deltas are reasonable
            if max_pos_delta > 0.5:  # 50cm threshold
                print(f"\n⚠️  WARNING: Large position deltas detected ({max_pos_delta:.4f} m)")
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
            
            # Interpolate trajectory to match control frequency
            # This ensures smooth movement even at slow speeds
            total_duration = timestamps[-1] - timestamps[0]
            # If slowing down, the duration is already extended because 'timestamps' were scaled by slow_down factor earlier
            
            num_steps = max(int(total_duration * control_frequency), len(poses))
            target_times = np.linspace(timestamps[0], timestamps[-1], num_steps)
            
            # Interpolate DELTAS at target times (not absolute poses)
            from scipy.interpolate import interp1d
            # Interpolate position deltas
            pos_delta_interp = interp1d(timestamps, deltas[:, :3], axis=0, kind='linear', 
                                       fill_value='extrapolate', assume_sorted=True)
            # Interpolate rotation deltas (as rotation vectors)
            rot_delta_interp = interp1d(timestamps, deltas[:, 3:], axis=0, kind='linear',
                                       fill_value='extrapolate', assume_sorted=True)
            
            interpolated_deltas = np.hstack([
                pos_delta_interp(target_times),
                rot_delta_interp(target_times)
            ])
            
            print(f"Interpolated {len(interpolated_deltas)} delta commands from {len(deltas)} trajectory points")
            print(f"Replay duration: {total_duration:.3f} seconds\n")
            
            # Reset start time
            t_start = time.time()
            
            for iter_idx in range(len(interpolated_deltas)):
                # Calculate cycle end time
                t_cycle_end = t_start + (iter_idx + 1) * control_dt
                
                # Calculate target time for this waypoint
                # Use relative time from start + current absolute time + lookahead
                rel_time = target_times[iter_idx] - target_times[0]
                this_t_target = t_start + rel_time + command_lookahead
                
                # Get interpolated delta for this step
                delta = interpolated_deltas[iter_idx]
                
                # Calculate target pose: robot_initial_pose + delta
                # Position: simple addition
                target_pos = robot_initial_pose[:3] + delta[:3]
                
                # Rotation: compose rotations (apply relative rotation to initial rotation)
                r_initial = st.Rotation.from_rotvec(robot_initial_pose[3:])
                r_delta = st.Rotation.from_rotvec(delta[3:])
                r_target = r_delta * r_initial  # Apply relative rotation: R_delta * R_initial
                target_rot = r_target.as_rotvec()
                
                # Combine into target pose
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
