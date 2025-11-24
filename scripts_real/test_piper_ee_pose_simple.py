#!/usr/bin/env python3
"""
Simple test script for Piper end-effector pose control.

This script directly uses PiperInterpolationController to test end-effector pose control
without requiring cameras or full UMI environment setup.

Usage:
    python scripts_real/test_piper_ee_pose_simple.py --can_name can0
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
    Test Piper end-effector pose control with different positions.
    """
    print("=" * 60)
    print("Piper End-Effector Pose Simple Test Script")
    print("=" * 60)
    print(f"CAN interface: {can_name}")
    print(f"Control frequency: {frequency} Hz")
    print("=" * 60)
    print("\nThis script will test the end-effector with different poses:")
    print("  1. Initial position (current)")
    print("  2. Move up (+20mm in Z)")
    print("  3. Move down (-20mm in Z, back to initial)")
    print("  4. Move forward (+20mm in Y)")
    print("  5. Move back (-20mm in Y, back to initial)")
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
        
        # Get initial pose
        initial_state = controller.get_state()
        initial_pose = initial_state['ActualTCPPose']
        print(f"\nInitial arm pose: {initial_pose}")
        print(f"  Position: [{initial_pose[0]:.4f}, {initial_pose[1]:.4f}, {initial_pose[2]:.4f}] m")
        print(f"  Rotation: [{initial_pose[3]:.4f}, {initial_pose[4]:.4f}, {initial_pose[5]:.4f}] rad")
        
        # Get initial gripper state
        initial_gripper = controller.get_gripper_state()
        print(f"\nInitial gripper position: {initial_gripper['gripper_position']:.4f} m ({initial_gripper['gripper_position']*1000:.2f} mm)")
        print(f"Initial gripper state code: {initial_gripper['gripper_state']}")
        
        # Test positions (relative to initial)
        test_poses = [
            ("Initial", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Move up (+20mm Z)", [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]),
            ("Move down (-20mm Z)", [0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
            ("Move forward (+20mm Y)", [0.0, 0.02, 0.0, 0.0, 0.0, 0.0]),
            ("Move back (-20mm Y)", [0.0, -0.02, 0.0, 0.0, 0.0, 0.0]),
        ]
        
        for test_name, offset in test_poses:
            print("\n" + "=" * 60)
            print(f"Testing: {test_name}")
            print("=" * 60)
            
            # Calculate target pose
            target_pose = initial_pose + np.array(offset)
            
            print(f"Target pose: {target_pose}")
            print(f"  Position: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}] m")
            print(f"  Rotation: [{target_pose[3]:.4f}, {target_pose[4]:.4f}, {target_pose[5]:.4f}] rad")
            
            # Schedule waypoint
            target_time = time.time() + 2.0  # 2 seconds to reach target
            controller.schedule_waypoint(
                pose=target_pose,
                target_time=target_time
            )
            
            print(f"Scheduled waypoint to reach at {target_time:.2f} s")
            
            # Monitor progress
            start_time = time.time()
            last_print_time = start_time
            print_interval = 0.2  # Print every 0.2 seconds
            
            while time.time() < target_time + 0.5:  # Wait a bit after target time
                current_time = time.time()
                
                if current_time - last_print_time >= print_interval:
                    state = controller.get_state()
                    current_pose = state['ActualTCPPose']
                    elapsed = current_time - start_time
                    
                    # Calculate error
                    pos_error = np.linalg.norm(current_pose[:3] - target_pose[:3])
                    rot_error = np.linalg.norm(current_pose[3:] - target_pose[3:])
                    
                    print(f"  [{elapsed:.2f}s] Position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}] m, "
                          f"Error: {pos_error*1000:.2f} mm, Rot error: {np.degrees(rot_error):.2f} deg")
                    
                    last_print_time = current_time
                
                precise_wait(0.01)  # 100 Hz monitoring
            
            # Final check
            final_state = controller.get_state()
            final_pose = final_state['ActualTCPPose']
            pos_error = np.linalg.norm(final_pose[:3] - target_pose[:3])
            rot_error = np.linalg.norm(final_pose[3:] - target_pose[3:])
            
            print(f"\nFinal pose: {final_pose}")
            print(f"Position error: {pos_error*1000:.2f} mm")
            print(f"Rotation error: {np.degrees(rot_error):.2f} deg")
            
            if pos_error < 0.005:  # 5mm tolerance
                print("✓ Position reached successfully!")
            else:
                print("⚠ Warning: Position error is larger than expected")
            
            # Wait a bit before next test
            time.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
        # Final state
        final_state = controller.get_state()
        final_pose = final_state['ActualTCPPose']
        print(f"\nFinal arm pose: {final_pose}")
        print(f"  Position: [{final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f}] m")
        print(f"  Rotation: [{final_pose[3]:.4f}, {final_pose[4]:.4f}, {final_pose[5]:.4f}] rad")
        
        final_gripper = controller.get_gripper_state()
        print(f"\nFinal gripper state:")
        print(f"  Position: {final_gripper['gripper_position']:.4f} m ({final_gripper['gripper_position']*1000:.2f} mm)")
        print(f"  Force: {final_gripper['gripper_force']:.4f} N·m")
        print(f"  State code: {final_gripper['gripper_state']}")
        print(f"  Timestamp: {final_gripper['gripper_timestamp']}")
        
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

