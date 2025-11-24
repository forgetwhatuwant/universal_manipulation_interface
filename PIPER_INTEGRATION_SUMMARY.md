# Piper Robot Integration Summary

## Overview
Successfully integrated Piper robot arm into UMI (Universal Manipulation Interface) following the same pattern as UR5 and Franka robots.

## Files Created

### 1. `umi/real_world/piper_interface.py`
Low-level wrapper for Piper SDK that handles unit conversions:
- **Position**: Converts 0.001 mm ↔ meters
- **Rotation**: Converts 0.001 degrees (Euler XYZ) ↔ radians (rotation vector)
- **Joint angles**: Converts 0.001 degrees ↔ radians

**Key Methods:**
- `get_ee_pose()` → Returns `np.array([x, y, z, rx, ry, rz])` in meters/radians
- `get_joint_positions()` → Returns `np.array([j0...j5])` in radians
- `get_joint_velocities()` → Returns velocities (currently zeros, can be computed from differences)
- `update_desired_ee_pose(pose)` → Sends cartesian command to robot

### 2. `umi/real_world/piper_interpolation_controller.py`
Main controller class following UMI pattern:
- Inherits from `multiprocessing.Process` for real-time control
- Uses `SharedMemoryQueue` for commands
- Uses `SharedMemoryRingBuffer` for state
- Implements trajectory interpolation using `PoseTrajectoryInterpolator`
- Supports `servoL()` and `schedule_waypoint()` commands
- Control frequency: 50 Hz (configurable, lower than UR5/Franka due to CAN bus)

## Files Modified

### 3. `umi/real_world/umi_env.py`
Added Piper support for single-arm environment:
```python
elif robot_type.startswith('piper'):
    robot = PiperInterpolationController(
        shm_manager=shm_manager,
        can_name=robot_ip,  # CAN interface name
        frequency=50,
        max_pos_speed=max_pos_speed,
        max_rot_speed=max_rot_speed,
        verbose=False,
        receive_latency=robot_obs_latency
    )
```

### 4. `umi/real_world/bimanual_umi_env.py`
Added Piper support for dual-arm environment:
```python
elif rc['robot_type'].startswith('piper'):
    this_robot = PiperInterpolationController(
        shm_manager=shm_manager,
        can_name=rc['robot_ip'],  # CAN interface name
        frequency=50,
        max_pos_speed=max_pos_speed*cube_diag,
        max_rot_speed=max_rot_speed*cube_diag,
        verbose=False,
        receive_latency=rc['robot_obs_latency']
    )
```

## Usage

### Single Arm Configuration
```python
from umi.real_world.umi_env import UmiEnv

env = UmiEnv(
    output_dir="./output",
    robot_ip="can0",  # CAN interface name for Piper
    gripper_ip="192.168.0.18",
    robot_type="piper",  # Use "piper" instead of "ur5" or "franka"
    frequency=20,
    max_pos_speed=0.25,
    max_rot_speed=0.6,
    robot_obs_latency=0.01,  # Higher latency for CAN bus
    robot_action_latency=0.1
)
```

### Dual Arm Configuration
```yaml
# example/eval_robots_config.yaml
{
  "robots": [
    {
      "robot_type": "piper",
      "robot_ip": "can0",  # CAN interface name
      "robot_obs_latency": 0.01,
      "robot_action_latency": 0.1,
      "tcp_offset": 0.091  # From piper_fk.py: d[5] = 91mm
    }
  ],
  "grippers": [
    {
      "gripper_ip": "192.168.0.18",
      "gripper_port": 1000,
      "gripper_obs_latency": 0.01,
      "gripper_action_latency": 0.1
    }
  ]
}
```

## Key Differences from UR5/Franka

| Feature | UR5 | Franka | Piper |
|---------|-----|--------|-------|
| **Communication** | TCP/IP (RTDE) | TCP/IP (zerorpc) | CAN bus |
| **Frequency** | 125-500 Hz | 200-1000 Hz | 50 Hz (recommended) |
| **robot_ip param** | IP address | IP address | CAN interface name ("can0") |
| **Units (native)** | meters/radians | meters/radians | 0.001mm / 0.001deg |
| **Rotation format** | Rotation vector | Rotation vector | Euler XYZ |

## Prerequisites

1. **CAN Bus Setup**: 
   - CAN module must be activated (see `piper_sdk/README.MD`)
   - Run: `bash piper_sdk/can_activate.sh can0 1000000`

2. **Piper SDK Installation**:
   ```bash
   pip install piper_sdk
   # OR install from local directory:
   cd piper_sdk
   pip install .
   ```

3. **Robot Configuration**:
   - Robot must be in slave mode for control
   - CAN interface must be properly configured

## Testing

To test the integration:

```python
from umi.real_world.piper_interface import PiperInterface

# Test low-level interface
piper = PiperInterface(can_name="can0")
pose = piper.get_ee_pose()
print(f"Current pose: {pose}")
joints = piper.get_joint_positions()
print(f"Current joints: {joints}")
piper.close()
```

## Notes

- **Control Frequency**: Set to 50 Hz by default (lower than UR5/Franka) due to CAN bus limitations
- **Latency**: Higher observation latency (0.01s) compared to UR5 (0.0001s) due to CAN bus
- **Joint Velocities**: Currently returns zeros. Can be computed from position differences if needed
- **TCP Offset**: Default 0.091m (91mm) from `piper_fk.py` kinematics
- **Motion Mode**: Uses MOVEL (linear motion) mode for cartesian control

## Integration Status

✅ PiperInterface wrapper created
✅ PiperInterpolationController created  
✅ Integrated into umi_env.py
✅ Integrated into bimanual_umi_env.py
⏳ Testing pending (requires physical robot and CAN setup)

