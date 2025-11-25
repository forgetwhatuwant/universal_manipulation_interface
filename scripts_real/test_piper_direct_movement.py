#!/usr/bin/env python3
"""
Direct test of Piper coordinate system using native SDK format.
This will help us understand the actual coordinate mapping.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import time
import click
from piper_sdk import C_PiperInterface_V2


@click.command()
@click.option('--can_name', default='can0', help='CAN interface name (e.g., can0)')
@click.option('--distance', default=20.0, type=float, help='Movement distance in mm (default: 20mm = 2cm)')
def main(can_name, distance):
    """Test Piper movements using native SDK coordinate system."""
    
    print("=" * 80)
    print("Piper Direct Movement Test (Native SDK Format)")
    print("=" * 80)
    
    # Initialize Piper SDK directly
    print(f"\nConnecting to Piper robot on {can_name}...")
    piper = C_PiperInterface_V2(can_name=can_name)
    piper.ConnectPort()
    
    while not piper.EnablePiper():
        time.sleep(0.01)
    
    try:
        # Get initial pose
        print("\nGetting initial pose...")
        end_pose_msg = piper.GetArmEndPoseMsgs()
        ep = end_pose_msg.end_pose
        
        # Convert from 0.001mm to mm
        x_mm = ep.X_axis / 1000.0
        y_mm = ep.Y_axis / 1000.0
        z_mm = ep.Z_axis / 1000.0
        rx_deg = ep.RX_axis / 1000.0
        ry_deg = ep.RY_axis / 1000.0
        rz_deg = ep.RZ_axis / 1000.0
        
        print(f"\nInitial Pose (Piper native format):")
        print(f"  X: {x_mm:.2f} mm")
        print(f"  Y: {y_mm:.2f} mm")
        print(f"  Z: {z_mm:.2f} mm")
        print(f"  RX: {rx_deg:.2f} deg")
        print(f"  RY: {ry_deg:.2f} deg")
        print(f"  RZ: {rz_deg:.2f} deg")
        
        initial_pos = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        
        print("\n" + "=" * 80)
        print("WARNING: Robot will move!")
        print("This will test movements in Piper's native X, Y, Z axes.")
        print("Make sure the robot has clear space to move.")
        print("Press Enter to continue or Ctrl+C to cancel.")
        print("=" * 80)
        input()
        
        factor = 1000  # Convert mm to 0.001mm
        
        # Test X-axis movement
        print(f"\n{'='*80}")
        print(f"Testing Piper X-axis movement (+{distance:.1f}mm)")
        print(f"{'='*80}")
        print(f"Position before: X={x_mm:.2f}, Y={y_mm:.2f}, Z={z_mm:.2f}")
        
        test_pos = initial_pos.copy()
        test_pos[0] += distance  # Increase X
        
        X = int(round(test_pos[0] * factor))
        Y = int(round(test_pos[1] * factor))
        Z = int(round(test_pos[2] * factor))
        RX = int(round(test_pos[3] * factor))
        RY = int(round(test_pos[4] * factor))
        RZ = int(round(test_pos[5] * factor))
        
        print(f"Sending command: X={test_pos[0]:.2f}mm")
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        
        print("Waiting 3 seconds...")
        time.sleep(3.0)
        
        # Read actual position
        end_pose_msg = piper.GetArmEndPoseMsgs()
        ep = end_pose_msg.end_pose
        x_after = ep.X_axis / 1000.0
        y_after = ep.Y_axis / 1000.0
        z_after = ep.Z_axis / 1000.0
        
        print(f"Position after:  X={x_after:.2f}, Y={y_after:.2f}, Z={z_after:.2f}")
        print(f"Movement: X={x_after-x_mm:.2f}mm, Y={y_after-y_mm:.2f}mm, Z={z_after-z_mm:.2f}mm")
        print("→ Observe which direction the robot moved!")
        
        # Return to initial
        print("\nReturning to initial pose...")
        X = int(round(initial_pos[0] * factor))
        Y = int(round(initial_pos[1] * factor))
        Z = int(round(initial_pos[2] * factor))
        RX = int(round(initial_pos[3] * factor))
        RY = int(round(initial_pos[4] * factor))
        RZ = int(round(initial_pos[5] * factor))
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        time.sleep(2.0)
        
        input("\nPress Enter to continue to Y-axis test...")
        
        # Test Y-axis movement
        print(f"\n{'='*80}")
        print(f"Testing Piper Y-axis movement (+{distance:.1f}mm)")
        print(f"{'='*80}")
        print(f"Position before: X={x_mm:.2f}, Y={y_mm:.2f}, Z={z_mm:.2f}")
        
        test_pos = initial_pos.copy()
        test_pos[1] += distance  # Increase Y
        
        X = int(round(test_pos[0] * factor))
        Y = int(round(test_pos[1] * factor))
        Z = int(round(test_pos[2] * factor))
        RX = int(round(test_pos[3] * factor))
        RY = int(round(test_pos[4] * factor))
        RZ = int(round(test_pos[5] * factor))
        
        print(f"Sending command: Y={test_pos[1]:.2f}mm")
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        
        print("Waiting 3 seconds...")
        time.sleep(3.0)
        
        # Read actual position
        end_pose_msg = piper.GetArmEndPoseMsgs()
        ep = end_pose_msg.end_pose
        x_after = ep.X_axis / 1000.0
        y_after = ep.Y_axis / 1000.0
        z_after = ep.Z_axis / 1000.0
        
        print(f"Position after:  X={x_after:.2f}, Y={y_after:.2f}, Z={z_after:.2f}")
        print(f"Movement: X={x_after-x_mm:.2f}mm, Y={y_after-y_mm:.2f}mm, Z={z_after-z_mm:.2f}mm")
        print("→ Observe which direction the robot moved!")
        
        # Return to initial
        print("\nReturning to initial pose...")
        X = int(round(initial_pos[0] * factor))
        Y = int(round(initial_pos[1] * factor))
        Z = int(round(initial_pos[2] * factor))
        RX = int(round(initial_pos[3] * factor))
        RY = int(round(initial_pos[4] * factor))
        RZ = int(round(initial_pos[5] * factor))
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        time.sleep(2.0)
        
        input("\nPress Enter to continue to Z-axis test...")
        
        # Test Z-axis movement (we know this moves up/down from demo)
        print(f"\n{'='*80}")
        print(f"Testing Piper Z-axis movement (+{distance:.1f}mm)")
        print(f"{'='*80}")
        print(f"Position before: X={x_mm:.2f}, Y={y_mm:.2f}, Z={z_mm:.2f}")
        print("Expected: Robot should move UP (we know this from demo)")
        
        test_pos = initial_pos.copy()
        test_pos[2] += distance  # Increase Z
        
        X = int(round(test_pos[0] * factor))
        Y = int(round(test_pos[1] * factor))
        Z = int(round(test_pos[2] * factor))
        RX = int(round(test_pos[3] * factor))
        RY = int(round(test_pos[4] * factor))
        RZ = int(round(test_pos[5] * factor))
        
        print(f"Sending command: Z={test_pos[2]:.2f}mm")
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        
        print("Waiting 3 seconds...")
        time.sleep(3.0)
        
        # Read actual position
        end_pose_msg = piper.GetArmEndPoseMsgs()
        ep = end_pose_msg.end_pose
        x_after = ep.X_axis / 1000.0
        y_after = ep.Y_axis / 1000.0
        z_after = ep.Z_axis / 1000.0
        
        print(f"Position after:  X={x_after:.2f}, Y={y_after:.2f}, Z={z_after:.2f}")
        print(f"Movement: X={x_after-x_mm:.2f}mm, Y={y_after-y_mm:.2f}mm, Z={z_after-z_mm:.2f}mm")
        print("→ Robot should have moved UP")
        
        # Return to initial
        print("\nReturning to initial pose...")
        X = int(round(initial_pos[0] * factor))
        Y = int(round(initial_pos[1] * factor))
        Z = int(round(initial_pos[2] * factor))
        RX = int(round(initial_pos[3] * factor))
        RY = int(round(initial_pos[4] * factor))
        RZ = int(round(initial_pos[5] * factor))
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        time.sleep(2.0)
        
        print("\n" + "=" * 80)
        print("Test Summary:")
        print("  Piper X-axis → ? direction")
        print("  Piper Y-axis → ? direction")
        print("  Piper Z-axis → UP direction (confirmed)")
        print("=" * 80)
        
    finally:
        piper.DisconnectPort()
        print("\nDisconnected from robot.")


if __name__ == '__main__':
    main()

