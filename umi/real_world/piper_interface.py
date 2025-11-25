import numpy as np
import scipy.spatial.transform as st
from piper_sdk import C_PiperInterface_V2
from umi.common.pose_util import pose_to_mat, mat_to_pose

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
        
        # Coordinate frame transformation: Piper J6 → UMI convention
        # 
        # CONFIRMED from direct movement tests:
        #   Piper X-axis → Forward/Backward movement
        #   Piper Y-axis → Left/Right movement
        #   Piper Z-axis → Up/Down movement ✓
        # 
        # UMI convention: X=right, Y=up, Z=forward
        # 
        # Position mapping:
        #   UMI X (right) = Piper Y (left/right) - sign to be verified
        #   UMI Y (up) = Piper Z (up/down) ✓ CONFIRMED
        #   UMI Z (forward) = Piper X (forward/backward) - sign to be verified
        # 
        # Initial mapping (signs may need adjustment based on actual movement direction):
        self.tx_piper_j6_to_umi = np.identity(4)
        self.tx_piper_j6_to_umi[:3, :3] = np.array([
            [0, 1, 0],   # UMI X (right) = Piper Y (left/right)
            [0, 0, 1],   # UMI Y (up) = Piper Z (up/down) ✓
            [1, 0, 0],   # UMI Z (forward) = Piper X (forward/backward)
        ])
        self.tx_umi_to_piper_j6 = np.linalg.inv(self.tx_piper_j6_to_umi)
    
    def get_ee_pose(self):
        """
        Get end-effector pose in UMI format.
        
        Returns:
            np.array([x, y, z, rx, ry, rz]) in meters/radians (rotation vector)
            Transformed from Piper J6 frame to UMI convention.
        """
        end_pose_msg = self.piper.GetArmEndPoseMsgs()
        ep = end_pose_msg.end_pose
        
        # Position: 0.001 mm → meters (Piper J6 frame)
        x_piper = ep.X_axis / 1e6
        y_piper = ep.Y_axis / 1e6
        z_piper = ep.Z_axis / 1e6
        
        # Rotation: 0.001 degrees → radians → rotation vector (Piper J6 frame)
        rx_deg = ep.RX_axis / 1000.0
        ry_deg = ep.RY_axis / 1000.0
        rz_deg = ep.RZ_axis / 1000.0
        
        # Convert Euler angles (XYZ intrinsic order) to rotation vector
        euler_rad = np.array([np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)])
        rot_piper = st.Rotation.from_euler('XYZ', euler_rad)
        
        # Convert to pose format and transform from Piper J6 to UMI convention
        pose_piper = np.array([x_piper, y_piper, z_piper, 
                               rot_piper.as_rotvec()[0], 
                               rot_piper.as_rotvec()[1], 
                               rot_piper.as_rotvec()[2]])
        pose_mat_piper = pose_to_mat(pose_piper)
        pose_mat_umi = self.tx_piper_j6_to_umi @ pose_mat_piper
        pose_umi = mat_to_pose(pose_mat_umi)
        
        return pose_umi
    
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
        Get joint velocities from Piper SDK high-speed feedback.
        
        Returns:
            np.array([v0, v1, v2, v3, v4, v5]) in rad/s
        """
        motor_info = self.piper.GetArmHighSpdInfoMsgs()
        
        # Extract motor speeds: units are 0.001 rad/s, convert to rad/s
        velocities = np.array([
            motor_info.motor_1.motor_speed / 1000.0,  # 0.001 rad/s → rad/s
            motor_info.motor_2.motor_speed / 1000.0,
            motor_info.motor_3.motor_speed / 1000.0,
            motor_info.motor_4.motor_speed / 1000.0,
            motor_info.motor_5.motor_speed / 1000.0,
            motor_info.motor_6.motor_speed / 1000.0
        ])
        
        return velocities
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        """
        Send desired end-effector pose command.
        
        Args:
            pose: np.array([x, y, z, rx, ry, rz]) in meters/radians (rotation vector)
                  In UMI convention, will be transformed to Piper J6 frame.
        """
        # Transform from UMI convention to Piper J6 frame
        pose_mat_umi = pose_to_mat(pose)
        pose_mat_piper = self.tx_umi_to_piper_j6 @ pose_mat_umi
        pose_piper = mat_to_pose(pose_mat_piper)
        
        # Position: meters → 0.001 mm (Piper J6 frame)
        X = int(round(pose_piper[0] * 1e6))
        Y = int(round(pose_piper[1] * 1e6))
        Z = int(round(pose_piper[2] * 1e6))
        
        # Rotation: rotation vector → Euler angles (XYZ) → 0.001 degrees (Piper J6 frame)
        rot_vec = pose_piper[3:]
        rot = st.Rotation.from_rotvec(rot_vec)
        euler_rad = rot.as_euler('XYZ')
        euler_deg = np.degrees(euler_rad)
        
        RX = int(round(euler_deg[0] * 1000.0))
        RY = int(round(euler_deg[1] * 1000.0))
        RZ = int(round(euler_deg[2] * 1000.0))
        
        # Set motion mode to MOVEP (point-to-point) for cartesian control
        # Mode: 0x01=CAN control, 0x00=MOVEP, speed=100, is_mit=0x00
        # Note: Matching demo script which uses MOVEP (0x00) instead of MOVEL (0x02)
        # MOVEP is sent every iteration to ensure mode is maintained
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        
        # Send pose command (in Piper J6 frame)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
    
    def get_gripper_position(self):
        """
        Get gripper position in UMI format.
        
        Returns:
            float: Gripper opening width in meters
        """
        gripper_msg = self.piper.GetArmGripperMsgs()
        # Convert from 0.001 mm to meters
        gripper_angle_um = gripper_msg.gripper_state.grippers_angle  # 0.001 mm
        gripper_position_m = gripper_angle_um / 1e6  # meters
        return gripper_position_m
    
    def get_gripper_state(self):
        """
        Get full gripper state in UMI format.
        
        Returns:
            dict: Gripper state with keys matching WSGController format
        """
        import time
        gripper_msg = self.piper.GetArmGripperMsgs()
        
        # Convert from 0.001 mm to meters
        gripper_angle_um = gripper_msg.gripper_state.grippers_angle  # 0.001 mm
        gripper_position_m = gripper_angle_um / 1e6  # meters
        
        # Convert effort from 0.001 N·m to N·m
        gripper_effort_nm = gripper_msg.gripper_state.grippers_effort / 1000.0
        
        # Extract status flags and parse FOC status
        status_code = gripper_msg.gripper_state.status_code
        foc_status = gripper_msg.gripper_state.foc_status
        
        return {
            'gripper_state': status_code,
            'gripper_position': gripper_position_m,  # meters
            'gripper_velocity': 0.0,  # Not directly available, could compute from differences
            'gripper_force': gripper_effort_nm,  # N·m
            'gripper_measure_timestamp': gripper_msg.time_stamp,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time() - 0.01,  # Approximate latency
            # Additional status info for debugging
            'driver_enabled': foc_status.driver_enable_status,
            'homing_status': foc_status.homing_status,
        }
    
    def update_desired_gripper_position(self, position: float, effort: float = 1000.0):
        """
        Send desired gripper position command.
        
        Args:
            position: Gripper opening width in meters (UMI uses 0 to 0.09m = 90mm typically)
            effort: Gripper effort in 0.001 N/m (default 1000 = 1 N/m)
        """
        # Convert meters → 0.001 mm
        gripper_angle_um = int(round(position * 1e6))
        
        # Get SDK gripper range limits if available, otherwise use hardware max (150mm)
        try:
            g_min, g_max = self.piper.GetSDKGripperRangeParam()
            g_min_um = int(round(g_min * 1e6))  # Convert meters to 0.001 mm
            g_max_um = int(round(g_max * 1e6))
            # Clamp to SDK limits
            gripper_angle_um = max(g_min_um, min(gripper_angle_um, g_max_um))
        except:
            # Fallback: clamp to hardware limits (0-150mm)
            gripper_angle_um = max(0, min(gripper_angle_um, 150000))  # 150mm max
        
        # Send gripper command
        # Use 0x03 (enable and clear errors) instead of 0x01 to handle error conditions
        # This matches the demo behavior of sending enable commands every iteration
        self.piper.GripperCtrl(
            gripper_angle=gripper_angle_um,
            gripper_effort=int(effort),  # 0.001 N/m
            gripper_code=0x03,  # Enable and clear errors (more robust than 0x01)
            set_zero=0x00  # Don't set zero
        )
    
    def close(self):
        """Close the Piper interface."""
        self.piper.DisconnectPort()

