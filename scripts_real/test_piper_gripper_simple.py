#!/usr/bin/env python3
"""
Simple test script for Piper gripper control.

This script directly uses PiperInterpolationController to test gripper functionality
without requiring cameras or full UMI environment setup.

Usage:
    python scripts_real/test_piper_gripper_simple.py --can_name can0
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
@click.option('--frequency', '-f', default=10, type=float, help='Control frequency in Hz')
def main(can_name, frequency):
    """
    Test Piper gripper control with different positions.
    """
    print("=" * 60)
    print("Piper Gripper Simple Test Script")
    print("=" * 60)
    print(f"CAN interface: {can_name}")
    print(f"Control frequency: {frequency} Hz")
    print("=" * 60)
    print("\nThis script will test the gripper with different positions:")
    print("  1. Close (0.0 m)")
    print("  2. Half open (0.025 m = 25 mm)")
    print("  3. Fully open (0.05 m = 50 mm)")
    print("  4. Close again (0.0 m)")
    print("\nPress Ctrl+C to stop at any time.")
    print("=" * 60)
    
    # Wait for user confirmation
    input("\nPress Enter to start the test...")
    
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    try:
        # Create Piper controller directly
        controller = PiperInterpolationController(
            shm_manager=shm_manager,
            can_name=can_name,
            frequency=frequency,
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            verbose=True
        )
        
        # Start controller
        print("\nStarting controller...")
        controller.start(wait=True)
        print("Controller started!")
        
        # Wait a bit for initialization
        time.sleep(2.0)
        
        # Get initial robot state to keep arm fixed
        initial_state = controller.get_state()
        initial_pose = initial_state['ActualTCPPose']
        print(f"\nInitial arm pose: {initial_pose}")
        print("Arm will remain fixed during gripper test.\n")
        
        # Get initial gripper state
        initial_gripper_state = controller.get_gripper_state()
        initial_gripper_pos = initial_gripper_state['gripper_position']
        print(f"Initial gripper position: {initial_gripper_pos:.4f} m ({initial_gripper_pos*1000:.2f} mm)")
        print(f"Initial gripper state code: {initial_gripper_state['gripper_state']}")
        print(f"Initial gripper force: {initial_gripper_state['gripper_force']:.4f} N·m")
        print(f"Driver enabled: {initial_gripper_state.get('driver_enabled', 'N/A')}")
        print(f"Homing status: {initial_gripper_state.get('homing_status', 'N/A')}")
        if not initial_gripper_state.get('driver_enabled', False):
            print("⚠ WARNING: Gripper driver is not enabled! Commands may not work.")
        print("\n" + "=" * 60)
        
        # Define test positions (in meters)
        # UMI typically uses 0 to 0.09m (90mm) range
        # Test with different positions within typical UMI range
        test_positions = [
            (0.0, "Close (0.0 m)"),
            (0.03, "Quarter open (0.03 m = 30 mm)"),
            (0.06, "Half open (0.06 m = 60 mm)"),
            (0.09, "Fully open (0.09 m = 90 mm)"),
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
            controller.schedule_waypoint(
                pose=initial_pose,
                target_time=target_time
            )
            
            # Schedule waypoint for gripper
            gripper_target_time = current_time + 1.0  # 1 second to reach position
            controller.schedule_gripper_waypoint(
                pos=pos_m,
                target_time=gripper_target_time
            )
            
            print(f"Scheduled gripper to move to {pos_m:.4f} m ({pos_m*1000:.2f} mm)")
            print(f"Target time: {gripper_target_time:.2f} s")
            
            # Monitor gripper movement
            start_time = time.time()
            duration = 3.0  # Monitor for 3 seconds
            last_pos = None
            last_print_time = start_time
            
            while time.time() - start_time < duration:
                # Get current gripper state
                gripper_state = controller.get_gripper_state()
                current_pos = gripper_state['gripper_position']
                current_force = gripper_state['gripper_force']
                current_state_code = gripper_state['gripper_state']
                
                # Print updates every 0.2 seconds or if position changed significantly
                elapsed = time.time() - start_time
                should_print = False
                if last_pos is None:
                    should_print = True
                elif abs(current_pos - last_pos) > 0.001:  # 1mm change
                    should_print = True
                elif time.time() - last_print_time > 0.2:  # Every 0.2s
                    should_print = True
                
                if should_print:
                    driver_enabled = gripper_state.get('driver_enabled', 'N/A')
                    print(f"  [{elapsed:.2f}s] Position: {current_pos:.4f} m ({current_pos*1000:.2f} mm), "
                          f"Force: {current_force:.4f} N·m, State: {current_state_code}, Enabled: {driver_enabled}")
                    last_pos = current_pos
                    last_print_time = time.time()
                
                precise_wait(start_time + dt, time_func=time.time)
            
            # Final state
            final_gripper_state = controller.get_gripper_state()
            final_pos = final_gripper_state['gripper_position']
            print(f"\nFinal gripper position: {final_pos:.4f} m ({final_pos*1000:.2f} mm)")
            print(f"Target was: {pos_m:.4f} m ({pos_m*1000:.2f} mm)")
            error_mm = abs(final_pos - pos_m) * 1000
            print(f"Error: {error_mm:.2f} mm")
            
            if error_mm < 2.0:
                print("✓ Position reached successfully!")
            else:
                print("⚠ Warning: Position error is larger than expected")
            
            # Wait before next test
            current_time = time.time() + 1.0
            time.sleep(1.0)
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
        # Final state summary
        final_state = controller.get_gripper_state()
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
        print("\nStopping controller...")
        controller.stop(wait=True)
        shm_manager.shutdown()
        print("Controller stopped.")

if __name__ == '__main__':
    main()

