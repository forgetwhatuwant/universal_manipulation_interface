import numpy as np
import scipy.spatial.transform as st
from piper_sdk import C_PiperInterface_V2

class PiperInterface:
    """
    Low-level wrapper for Piper SDK to match UMI interface pattern.
    Handles unit conversions between Piper SDK format and UMI standard format.
    """
    
    def __init__(self, can_name="can0", dh_is_offset=0x01):
        """
        Initialize Piper interface.
        
        Args:
            can_name: CAN interface name (e.g., "can0", "can1")
            dh_is_offset: DH parameter offset flag (0x00 or 0x01)
        """
        self.piper = C_PiperInterface_V2(
            can_name=can_name,
            judge_flag=False,  # Don't validate CAN port
            can_auto_init=True,
            dh_is_offset=dh_is_offset,
            start_sdk_joint_limit=False,
            start_sdk_gripper_limit=False
        )
        self.piper.ConnectPort()
        
        # Wait for robot to be enabled
        import time
        while not self.piper.EnablePiper():
            time.sleep(0.01)
    
    def get_ee_pose(self):
        """
        Get end-effector pose in UMI format.
        
        Returns:
            np.array([x, y, z, rx, ry, rz]) in meters/radians (rotation vector)
        """
        end_pose_msg = self.piper.GetArmEndPoseMsgs()
        ep = end_pose_msg.end_pose
        
        # Position: 0.001 mm → meters
        x = ep.X_axis / 1e6
        y = ep.Y_axis / 1e6
        z = ep.Z_axis / 1e6
        
        # Rotation: 0.001 degrees → radians → rotation vector
        rx_deg = ep.RX_axis / 1000.0
        ry_deg = ep.RY_axis / 1000.0
        rz_deg = ep.RZ_axis / 1000.0
        
        # Convert Euler angles (XYZ intrinsic order) to rotation vector
        euler_rad = np.array([np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)])
        rot = st.Rotation.from_euler('XYZ', euler_rad)
        rot_vec = rot.as_rotvec()
        
        return np.array([x, y, z, rot_vec[0], rot_vec[1], rot_vec[2]])
    
    def get_joint_positions(self):
        """
        Get joint positions in UMI format.
        
        Returns:
            np.array([j0, j1, j2, j3, j4, j5]) in radians
        """
        joint_msg = self.piper.GetArmJointMsgs()
        js = joint_msg.joint_state
        
        # Joint angles: 0.001 degrees → radians
        # Conversion factor: 57295.7795 = 1000 * 180 / π
        factor = 57295.7795
        joints = np.array([
            js.joint_1 / factor,
            js.joint_2 / factor,
            js.joint_3 / factor,
            js.joint_4 / factor,
            js.joint_5 / factor,
            js.joint_6 / factor
        ])
        
        return joints
    
    def get_joint_velocities(self):
        """
        Get joint velocities. 
        Note: Piper SDK may not provide direct velocity feedback.
        This would need to be computed from position differences.
        
        Returns:
            np.array([v0, v1, v2, v3, v4, v5]) in rad/s
        """
        # Piper SDK doesn't seem to provide direct velocity feedback
        # Return zeros for now, could be computed from position differences
        return np.zeros(6)
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        """
        Send desired end-effector pose command.
        
        Args:
            pose: np.array([x, y, z, rx, ry, rz]) in meters/radians (rotation vector)
        """
        # Position: meters → 0.001 mm
        X = int(round(pose[0] * 1e6))
        Y = int(round(pose[1] * 1e6))
        Z = int(round(pose[2] * 1e6))
        
        # Rotation: rotation vector → Euler angles (XYZ) → 0.001 degrees
        rot_vec = pose[3:]
        rot = st.Rotation.from_rotvec(rot_vec)
        euler_rad = rot.as_euler('XYZ')
        euler_deg = np.degrees(euler_rad)
        
        RX = int(round(euler_deg[0] * 1000.0))
        RY = int(round(euler_deg[1] * 1000.0))
        RZ = int(round(euler_deg[2] * 1000.0))
        
        # Set motion mode to MOVEL (linear motion) for cartesian control
        # Mode: 0x01=CAN control, 0x02=MOVEL, speed=100, is_mit=0x00
        self.piper.MotionCtrl_2(0x01, 0x02, 100, 0x00)
        
        # Send pose command
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
    
    def close(self):
        """Close the Piper interface."""
        self.piper.DisconnectPort()

