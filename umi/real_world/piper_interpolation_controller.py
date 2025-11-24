import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
from umi.real_world.piper_interface import PiperInterface

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    SCHEDULE_GRIPPER_WAYPOINT = 3


class PiperInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        can_name="can0",
        frequency=100,  # Control frequency in Hz (100 Hz recommended, up to 200 Hz supported)
        max_pos_speed=0.25,  # m/s
        max_rot_speed=0.6,  # rad/s
        launch_timeout=3,
        joints_init=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0
        ):
        """
        robot_can_name: the CAN interface name (e.g., "can0")
        frequency: Control frequency in Hz (100 Hz recommended, up to 200 Hz supported)
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """
        
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="PiperPositionalController")
        self.can_name = can_name
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue (for robot pose commands)
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )
        
        # build gripper input queue
        gripper_example = {
            'cmd': Command.SCHEDULE_GRIPPER_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        gripper_input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=gripper_example,
            buffer_size=256
        )

        # build ring buffer (for robot state)
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd', 'get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(6)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        # build gripper ring buffer (matching WSGController format)
        gripper_example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        gripper_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=gripper_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.gripper_input_queue = gripper_input_queue
        self.ring_buffer = ring_buffer
        self.gripper_ring_buffer = gripper_ring_buffer
        self.receive_keys = receive_keys
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PiperPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        self.gripper_input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    def schedule_gripper_waypoint(self, pos: float, target_time: float):
        """
        Schedule a gripper waypoint.
        
        Args:
            pos: Gripper opening width in meters
            target_time: Absolute time when gripper should reach this position
        """
        message = {
            'cmd': Command.SCHEDULE_GRIPPER_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.gripper_input_queue.put(message)
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def get_gripper_state(self, k=None, out=None):
        """Get gripper state from ring buffer."""
        if k is None:
            return self.gripper_ring_buffer.get(out=out)
        else:
            return self.gripper_ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_gripper_state(self):
        return self.gripper_ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # start piper interface
        robot = PiperInterface(can_name=self.can_name)

        try:
            if self.verbose:
                print(f"[PiperPositionalController] Connect to robot: {self.can_name}")
            
            # init pose (if needed)
            # Note: Piper doesn't have a direct move_to_joint_positions like Franka
            # Joint initialization would need to be done via JointCtrl if needed
            
            # Initialize gripper (enable it) - matching demo script sequence exactly
            # Disable and clear errors first
            robot.piper.GripperCtrl(0, 1000, 0x02, 0)
            time.sleep(0.01)
            
            # Get current gripper position before enabling
            initial_gripper_msg = robot.piper.GetArmGripperMsgs()
            initial_gripper_pos_um = initial_gripper_msg.gripper_state.grippers_angle
            
            # Enable gripper - send enable commands continuously at 200 Hz like the demo
            # The demo sends commands every 0.005s (200 Hz) continuously
            enable_start = time.time()
            enable_timeout = 1.0  # Try for 1 second
            enable_success = False
            enable_iterations = 0
            max_enable_iterations = int(enable_timeout / 0.005)  # 200 Hz for 1 second
            
            while enable_iterations < max_enable_iterations:
                # Send enable command with current gripper position (use abs like demo)
                robot.piper.GripperCtrl(abs(initial_gripper_pos_um), 1000, 0x01, 0)
                time.sleep(0.005)  # 200 Hz like the demo
                enable_iterations += 1
                
                # Check status every 10 iterations (every 50ms) to avoid excessive polling
                if enable_iterations % 10 == 0:
                    gripper_state_check = robot.get_gripper_state()
                    if gripper_state_check.get('driver_enabled', False):
                        enable_success = True
                        if self.verbose:
                            print(f"[PiperPositionalController] Gripper driver enabled successfully after {enable_iterations * 0.005:.3f}s!")
                        break
            
            if not enable_success:
                gripper_state_check = robot.get_gripper_state()
                if self.verbose:
                    print(f"[PiperPositionalController] WARNING: Gripper driver not enabled after {enable_timeout}s!")
                    print(f"[PiperPositionalController] Status code: {gripper_state_check['gripper_state']} (0x{gripper_state_check['gripper_state']:02x})")
                    print(f"[PiperPositionalController] FOC Status: driver_enabled={gripper_state_check.get('driver_enabled', False)}, homing_status={gripper_state_check.get('homing_status', False)}")
                    print(f"[PiperPositionalController] Gripper position: {gripper_state_check['gripper_position']*1000:.2f} mm")
                    # Print all FOC status flags for debugging
                    foc_status = initial_gripper_msg.gripper_state.foc_status
                    print(f"[PiperPositionalController] All FOC flags: voltage_too_low={foc_status.voltage_too_low}, "
                          f"motor_overheating={foc_status.motor_overheating}, driver_overcurrent={foc_status.driver_overcurrent}, "
                          f"driver_overheating={foc_status.driver_overheating}, sensor_status={foc_status.sensor_status}, "
                          f"driver_error_status={foc_status.driver_error_status}")
            
            # Get final gripper state for initialization
            gripper_state = robot.get_gripper_state()
            
            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()
            
            # Get initial gripper position from state (more reliable than get_gripper_position)
            curr_gripper_pos = gripper_state['gripper_position']
            if self.verbose:
                print(f"[PiperPositionalController] Initial gripper position: {curr_gripper_pos:.4f} m ({curr_gripper_pos*1000:.2f} mm)")
                print(f"[PiperPositionalController] Gripper driver enabled: {gripper_state.get('driver_enabled', False)}")
                print(f"[PiperPositionalController] Gripper homing status: {gripper_state.get('homing_status', False)}")

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            last_gripper_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            # Gripper interpolator (using PoseTrajectoryInterpolator with only position)
            # Initialize with current gripper position
            gripper_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_gripper_pos, 0, 0, 0, 0, 0]]  # Only position matters
            )
            
            # Send initial gripper command to ensure it's at the current position
            # This matches the demo behavior of sending commands every iteration
            robot.update_desired_gripper_position(curr_gripper_pos)

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)

                # send command to robot
                robot.update_desired_ee_pose(pose_command)
                
                # send gripper command
                gripper_command = gripper_interp(t_now)
                gripper_pos = gripper_command[0]  # Extract position from pose array
                robot.update_desired_gripper_position(gripper_pos)

                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                    
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)
                
                # update gripper state
                gripper_state = robot.get_gripper_state()
                # Filter to only include fields that match WSGController format (exclude driver_enabled, homing_status)
                gripper_state_filtered = {
                    'gripper_state': gripper_state['gripper_state'],
                    'gripper_position': gripper_state['gripper_position'],
                    'gripper_velocity': gripper_state['gripper_velocity'],
                    'gripper_force': gripper_state['gripper_force'],
                    'gripper_measure_timestamp': gripper_state['gripper_measure_timestamp'],
                    'gripper_receive_timestamp': gripper_state['gripper_receive_timestamp'],
                    'gripper_timestamp': gripper_state['gripper_timestamp']
                }
                self.gripper_ring_buffer.put(gripper_state_filtered)

                # fetch command from queue
                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[PiperPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # fetch gripper commands from queue
                try:
                    gripper_commands = self.gripper_input_queue.get_k(1)
                    n_gripper_cmd = len(gripper_commands['cmd'])
                except Empty:
                    n_gripper_cmd = 0

                # execute gripper commands
                for i in range(n_gripper_cmd):
                    gripper_command = dict()
                    for key, value in gripper_commands.items():
                        gripper_command[key] = value[i]
                    gripper_cmd = gripper_command['cmd']

                    if gripper_cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif gripper_cmd == Command.SCHEDULE_GRIPPER_WAYPOINT.value:
                        target_gripper_pos = float(gripper_command['target_pos'])
                        target_time = float(gripper_command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        # Use schedule_waypoint with only position (gripper is 1D)
                        gripper_pose = np.array([target_gripper_pos, 0, 0, 0, 0, 0])
                        gripper_interp = gripper_interp.schedule_waypoint(
                            pose=gripper_pose,
                            time=target_time,
                            max_pos_speed=0.1,  # Max gripper speed (m/s)
                            max_rot_speed=np.inf,  # No rotation
                            curr_time=curr_time,
                            last_waypoint_time=last_gripper_waypoint_time
                        )
                        last_gripper_waypoint_time = target_time

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[PiperPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # mandatory cleanup
            robot.close()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[PiperPositionalController] Disconnected from robot: {self.can_name}")

