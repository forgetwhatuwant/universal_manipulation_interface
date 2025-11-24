#!/usr/bin/env python3
"""
Test script for Piper gripper control using UMI.

This script tests the Piper gripper independently by:
1. Keeping the arm pose fixed
2. Testing different gripper positions
3. Reading and displaying gripper state feedback

Usage:
    python scripts_real/test_piper_gripper.py --can_name can0
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import click
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.umi_env import UmiEnv
from umi.common.precise_sleep import precise_wait

@click.command()
@click.option('--can_name', default='can0', help='CAN interface name (e.g., can0)')
@click.option('--output_dir', '-o', default='data/test_piper_gripper', help='Output directory for recordings')
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz')
def main(can_name, output_dir, frequency):
    """
    Test Piper gripper control with different positions.
    """
    print("=" * 60)
    print("Piper Gripper Test Script")
    print("=" * 60)
    print(f"CAN interface: {can_name}")
    print(f"Control frequency: {frequency} Hz")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print("\nThis script will test the gripper with different positions:")
    print("  1. Close (0.0 m)")
    print("  2. Half open (0.025 m)")
    print("  3. Fully open (0.05 m)")
    print("  4. Close again (0.0 m)")
    print("\nPress Ctrl+C to stop at any time.")
    print("=" * 60)
    
    # Wait for user confirmation
    input("\nPress Enter to start the test...")
    
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    try:
        # Create UMI environment with Piper robot
        # Note: gripper_ip is not used for Piper, but required by UmiEnv
        env = UmiEnv(
            output_dir=output_dir,
            robot_ip=can_name,  # For Piper, this is the CAN interface name
            gripper_ip="dummy",  # Not used for Piper
            gripper_port=1000,  # Not used for Piper
            robot_type='piper',
            frequency=frequency,
            robot_obs_latency=0.01,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            shm_manager=shm_manager
        )
        
        # Start environment
        print("\nStarting environment...")
        env.start(wait=True)
        print("Environment started!")
        
        # Wait a bit for initialization
        time.sleep(1.0)
        
        # Get initial robot state to keep arm fixed
        initial_state = env.get_robot_state()
        initial_pose = initial_state['ActualTCPPose']
        print(f"\nInitial arm pose: {initial_pose}")
        print("Arm will remain fixed during gripper test.\n")
        
        # Get initial gripper state
        initial_gripper_state = env.robot.get_gripper_state()
        initial_gripper_pos = initial_gripper_state['gripper_position']
        print(f"Initial gripper position: {initial_gripper_pos:.4f} m ({initial_gripper_pos*1000:.2f} mm)")
        print(f"Initial gripper state: {initial_gripper_state['gripper_state']}")
        print(f"Initial gripper force: {initial_gripper_state['gripper_force']:.4f} N·m")
        print("\n" + "=" * 60)
        
        # Define test positions (in meters)
        test_positions = [
            (0.0, "Close (0.0 m)"),
            (0.025, "Half open (0.025 m = 25 mm)"),
            (0.05, "Fully open (0.05 m = 50 mm)"),
            (0.0, "Close again (0.0 m)")
        ]
        
        dt = 1.0 / frequency
        current_time = time.time()
        
        for pos_m, description in test_positions:
            print(f"\n{'='*60}")
            print(f"Testing: {description}")
            print(f"{'='*60}")
            
            # Schedule waypoint for arm (keep fixed)
            target_time = current_time + 0.5  # Small delay
            env.robot.schedule_waypoint(
                pose=initial_pose,
                target_time=target_time
            )
            
            # Schedule waypoint for gripper
            gripper_target_time = current_time + 1.0  # 1 second to reach position
            env.robot.schedule_gripper_waypoint(
                pos=pos_m,
                target_time=gripper_target_time
            )
            
            print(f"Scheduled gripper to move to {pos_m:.4f} m ({pos_m*1000:.2f} mm)")
            print(f"Target time: {gripper_target_time:.2f} s")
            
            # Monitor gripper movement
            start_time = time.time()
            duration = 3.0  # Monitor for 3 seconds
            last_pos = None
            
            while time.time() - start_time < duration:
                # Get current gripper state
                gripper_state = env.robot.get_gripper_state()
                current_pos = gripper_state['gripper_position']
                current_force = gripper_state['gripper_force']
                current_state_code = gripper_state['gripper_state']
                
                # Print if position changed significantly
                if last_pos is None or abs(current_pos - last_pos) > 0.001:
                    elapsed = time.time() - start_time
                    print(f"  [{elapsed:.2f}s] Position: {current_pos:.4f} m ({current_pos*1000:.2f} mm), "
                          f"Force: {current_force:.4f} N·m, State: {current_state_code}")
                    last_pos = current_pos
                
                precise_wait(start_time + dt, time_func=time.time)
            
            # Final state
            final_gripper_state = env.robot.get_gripper_state()
            final_pos = final_gripper_state['gripper_position']
            print(f"\nFinal gripper position: {final_pos:.4f} m ({final_pos*1000:.2f} mm)")
            print(f"Target was: {pos_m:.4f} m ({pos_m*1000:.2f} mm)")
            print(f"Error: {abs(final_pos - pos_m)*1000:.2f} mm")
            
            # Wait before next test
            current_time = time.time() + 1.0
            time.sleep(1.0)
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
        # Final state summary
        final_state = env.robot.get_gripper_state()
        print(f"\nFinal gripper state:")
        print(f"  Position: {final_state['gripper_position']:.4f} m ({final_state['gripper_position']*1000:.2f} mm)")
        print(f"  Force: {final_state['gripper_force']:.4f} N·m")
        print(f"  State code: {final_state['gripper_state']}")
        print(f"  Timestamp: {final_state['gripper_timestamp']:.4f}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nStopping environment...")
        env.stop(wait=True)
        shm_manager.shutdown()
        print("Environment stopped.")

if __name__ == '__main__':
    main()

