#!/usr/bin/env python3
"""
Debug script to check what poses are being sent to Piper.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from umi.real_world.piper_interface import PiperInterface
from umi.common.pose_util import pose_to_mat, mat_to_pose

# Test pose in UMI convention
test_pose_umi = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move 2cm in X

print("=" * 80)
print("Debugging Piper Transformation")
print("=" * 80)

print(f"\nInput pose (UMI convention):")
print(f"  Position: [{test_pose_umi[0]:.4f}, {test_pose_umi[1]:.4f}, {test_pose_umi[2]:.4f}] m")
print(f"  Rotation: [{test_pose_umi[3]:.4f}, {test_pose_umi[4]:.4f}, {test_pose_umi[5]:.4f}] rad")

# Create interface to get transformation matrix
piper = PiperInterface(can_name='can0')
try:
    # Get current pose
    current_pose_umi = piper.get_ee_pose()
    print(f"\nCurrent pose (UMI convention):")
    print(f"  Position: [{current_pose_umi[0]:.4f}, {current_pose_umi[1]:.4f}, {current_pose_umi[2]:.4f}] m")
    
    # Test transformation
    print(f"\nTransformation matrix (Piper J6 → UMI):")
    print(piper.tx_piper_j6_to_umi[:3, :3])
    
    print(f"\nInverse transformation matrix (UMI → Piper J6):")
    print(piper.tx_umi_to_piper_j6[:3, :3])
    
    # Transform test pose
    pose_mat_umi = pose_to_mat(test_pose_umi)
    pose_mat_piper = piper.tx_umi_to_piper_j6 @ pose_mat_umi
    pose_piper = mat_to_pose(pose_mat_piper)
    
    print(f"\nTransformed pose (Piper J6 frame):")
    print(f"  Position: [{pose_piper[0]:.4f}, {pose_piper[1]:.4f}, {pose_piper[2]:.4f}] m")
    print(f"  Position: [{pose_piper[0]*1000:.2f}, {pose_piper[1]*1000:.2f}, {pose_piper[2]*1000:.2f}] mm")
    print(f"  Rotation: [{pose_piper[3]:.4f}, {pose_piper[4]:.4f}, {pose_piper[5]:.4f}] rad")
    
    # Check if transformation is correct
    print(f"\nVerification:")
    print(f"  UMI X = {test_pose_umi[0]:.4f} m")
    print(f"  Should map to Piper: {piper.tx_umi_to_piper_j6[:3, :3] @ np.array([1, 0, 0])}")
    
finally:
    piper.close()

