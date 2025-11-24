#!/usr/bin/env python3
"""
Combined test script for Piper end-effector pose and gripper control.

This script tests both end-effector pose control and gripper control simultaneously
using PiperInterpolationController.

Usage:
    python scripts_real/test_piper_combined.py --can_name can0
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
from umi.real_world.piper_interpolation_controller import PiperInterpolationController
from umi.common.precise_sleep import precise_wait

@click.command()
@click.option('--can_name', default='can0', help='CAN interface name (e.g., can0)')
@click.option('--frequency', '-f', default=100, type=float, help='Control frequency in Hz')
def main(can_name, frequency):
    """
    Test Piper end-effector pose and gripper control simultaneously.
    """
    print("=" * 60)
    print("Piper Combined Test Script (EE Pose + Gripper)")
    print("=" * 60)
    print(f"CAN interface: {can_name}")
    print(f"Control frequency: {frequency} Hz")
    print("=" * 60)
    print("\nThis script will test both end-effector and gripper simultaneously:")
    print("  1. Move up (+20mm Z) while opening gripper (0.03 m)")
    print("  2. Move forward (+20mm Y) while closing gripper (0.0 m)")
    print("  3. Move down (-20mm Z) while opening gripper (0.06 m)")
    print("  4. Move back (-20mm Y) while closing gripper (0.0 m)")
    print("\nPress Ctrl+C to stop at any time.")
    print("=" * 60)
    
    # Wait for user confirmation
    input("\nPress Enter to start the test...")
    
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    try:
        # Create controller
        print("\nStarting controller...")
        controller = PiperInterpolationController(
            shm_manager=shm_manager,
            can_name=can_name,
            frequency=frequency,
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            verbose=True
        )
        controller.start()
        print("Controller started!")
        
        # Wait for controller to be ready
        time.sleep(1.0)
        
        # Get initial pose and gripper state
        initial_state = controller.get_state()
        initial_pose = initial_state['ActualTCPPose']
        initial_gripper = controller.get_gripper_state()
        
        print(f"\nInitial arm pose: {initial_pose}")
        print(f"  Position: [{initial_pose[0]:.4f}, {initial_pose[1]:.4f}, {initial_pose[2]:.4f}] m")
        print(f"  Rotation: [{initial_pose[3]:.4f}, {initial_pose[4]:.4f}, {initial_pose[5]:.4f}] rad")
        print(f"\nInitial gripper position: {initial_gripper['gripper_position']:.4f} m ({initial_gripper['gripper_position']*1000:.2f} mm)")
        print(f"Initial gripper state code: {initial_gripper['gripper_state']}")
        
        # Test sequences: (test_name, pose_offset, gripper_target)
        test_sequences = [
            ("Move up + Open gripper", [0.0, 0.0, 0.02, 0.0, 0.0, 0.0], 0.03),
            ("Move forward + Close gripper", [0.0, 0.02, 0.0, 0.0, 0.0, 0.0], 0.0),
            ("Move down + Open gripper", [0.0, 0.0, -0.02, 0.0, 0.0, 0.0], 0.06),
            ("Move back + Close gripper", [0.0, -0.02, 0.0, 0.0, 0.0, 0.0], 0.0),
        ]
        
        for test_name, pose_offset, gripper_target in test_sequences:
            print("\n" + "=" * 60)
            print(f"Testing: {test_name}")
            print("=" * 60)
            
            # Calculate target pose
            target_pose = initial_pose + np.array(pose_offset)
            
            print(f"Target pose: {target_pose}")
            print(f"  Position: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}] m")
            print(f"  Rotation: [{target_pose[3]:.4f}, {target_pose[4]:.4f}, {target_pose[5]:.4f}] rad")
            print(f"Target gripper position: {gripper_target:.4f} m ({gripper_target*1000:.2f} mm)")
            
            # Schedule waypoint for arm
            target_time = time.time() + 3.0  # 3 seconds to reach target
            controller.schedule_waypoint(
                pose=target_pose,
                target_time=target_time
            )
            
            # Schedule gripper waypoint
            controller.schedule_gripper_waypoint(
                pos=gripper_target,
                target_time=target_time
            )
            
            print(f"Scheduled waypoints to reach at {target_time:.2f} s")
            
            # Monitor progress
            start_time = time.time()
            last_print_time = start_time
            print_interval = 0.2  # Print every 0.2 seconds
            
            while time.time() < target_time + 0.5:  # Wait a bit after target time
                current_time = time.time()
                
                if current_time - last_print_time >= print_interval:
                    state = controller.get_state()
                    gripper_state = controller.get_gripper_state()
                    current_pose = state['ActualTCPPose']
                    current_gripper_pos = gripper_state['gripper_position']
                    elapsed = current_time - start_time
                    
                    # Calculate errors
                    pos_error = np.linalg.norm(current_pose[:3] - target_pose[:3])
                    rot_error = np.linalg.norm(current_pose[3:] - target_pose[3:])
                    gripper_error = abs(current_gripper_pos - gripper_target)
                    
                    print(f"  [{elapsed:.2f}s] Arm: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}] m, "
                          f"Error: {pos_error*1000:.2f} mm | "
                          f"Gripper: {current_gripper_pos*1000:.2f} mm, Error: {gripper_error*1000:.2f} mm")
                    
                    last_print_time = current_time
                
                precise_wait(0.01)  # 100 Hz monitoring
            
            # Final check
            final_state = controller.get_state()
            final_gripper = controller.get_gripper_state()
            final_pose = final_state['ActualTCPPose']
            final_gripper_pos = final_gripper['gripper_position']
            
            pos_error = np.linalg.norm(final_pose[:3] - target_pose[:3])
            rot_error = np.linalg.norm(final_pose[3:] - target_pose[3:])
            gripper_error = abs(final_gripper_pos - gripper_target)
            
            print(f"\nFinal arm pose: {final_pose}")
            print(f"Arm position error: {pos_error*1000:.2f} mm")
            print(f"Arm rotation error: {np.degrees(rot_error):.2f} deg")
            print(f"Final gripper position: {final_gripper_pos:.4f} m ({final_gripper_pos*1000:.2f} mm)")
            print(f"Gripper error: {gripper_error*1000:.2f} mm")
            
            if pos_error < 0.005 and gripper_error < 0.005:  # 5mm tolerance
                print("✓ Both arm and gripper reached target successfully!")
            else:
                if pos_error >= 0.005:
                    print("⚠ Warning: Arm position error is larger than expected")
                if gripper_error >= 0.005:
                    print("⚠ Warning: Gripper error is larger than expected")
            
            # Wait a bit before next test
            time.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
        # Final state
        final_state = controller.get_state()
        final_pose = final_state['ActualTCPPose']
        final_gripper = controller.get_gripper_state()
        print(f"\nFinal arm pose: {final_pose}")
        print(f"  Position: [{final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f}] m")
        print(f"  Rotation: [{final_pose[3]:.4f}, {final_pose[4]:.4f}, {final_pose[5]:.4f}] rad")
        print(f"\nFinal gripper state:")
        print(f"  Position: {final_gripper['gripper_position']:.4f} m ({final_gripper['gripper_position']*1000:.2f} mm)")
        print(f"  Force: {final_gripper['gripper_force']:.4f} N·m")
        print(f"  State code: {final_gripper['gripper_state']}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nStopping controller...")
        if 'controller' in locals():
            controller.stop()
            controller.join()
        print("Controller stopped.")
        shm_manager.shutdown()

if __name__ == "__main__":
    main()

