import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import re
import numpy as np
import pybullet as pyb  # pytype: disable=import-error

from motion_imitation.robots import laikago_constants
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.robots import solo12_kinematic

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "motor_abduction_fl", "motor_hip_fl", "motor_knee_fl",
    "motor_abduction_hr", "motor_hip_hr", "motor_knee_hr",
    "motor_abduction_fr", "motor_hip_fr", "motor_knee_fr",
    "motor_abduction_hl", "motor_hip_hl", "motor_knee_hl",
]

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.32]
JOINT_DIRECTIONS = np.ones(12)
ABDUCTION_JOINT_OFFSET = 0.0
HIP_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3

JOINT_OFFSETS = np.array(
    [ABDUCTION_JOINT_OFFSET, HIP_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2

# update hip default positions
_DEFAULT_HIP_POSITIONS = (
    (0.1946, -0.1015, 0),
    (0.1946, 0.1015, 0),
    (-0.1946, -0.1015, 0),
    (-0.1946, 0.1015, 0),
)

# change com_offset
COM_OFFSET = -np.array([-0.00082966, 0.00000105, -0.00060210])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0.79, -1.7] * NUM_LEGS)

URDF_FILENAME = "motion_imitation/robots/simulation/solo12_new.urdf"

def analytical_leg_jacobian(leg_angles, leg_id):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
      l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1) ** (leg_id + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(
        t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J


def foot_positions_in_base_frame(foot_angles):
    foot_angles = foot_angles.reshape((4, 3))
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                       l_hip_sign=(-1) ** (i + 1))
    return foot_positions + HIP_OFFSETS


class Solo12(minitaur.Minitaur):
    """A simulation for the Solo12 robot."""
    def __init__(
            self,
            pybullet_client,
            urdf_filename=URDF_FILENAME,
            enable_clip_motor_commands=False,
            time_step=0.001,
            action_repeat=10,
            sensors=None,
            control_latency=0.002,
            on_rack=False,
            enable_action_interpolation=True,
            enable_action_filter=False,
            motor_control_mode=None,
            reset_time=1,
            allow_knee_contact=False,
    ):
        self._urdf_filename = urdf_filename
        self._allow_knee_contact = allow_knee_contact
        self._enable_clip_motor_commands = enable_clip_motor_commands
        self.plane = None


        motor_kp = [
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
            HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
        ]
        motor_kd = [
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
            HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
        ]

        super().__init__(
            pybullet_client=pybullet_client,
            time_step=time_step,
            action_repeat=action_repeat,
            num_motors=NUM_MOTORS,
            dofs_per_leg=DOFS_PER_LEG,
            motor_direction=JOINT_DIRECTIONS,
            motor_offset=JOINT_OFFSETS,
            motor_overheat_protection=False,
            motor_control_mode=motor_control_mode,
            motor_model_class=laikago_motor.LaikagoMotorModel,
            sensors=sensors,
            motor_kp=motor_kp,
            motor_kd=motor_kd,
            control_latency=control_latency,
            on_rack=on_rack,
            enable_action_interpolation=enable_action_interpolation,
            enable_action_filter=enable_action_filter,
            reset_time=reset_time)

    def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
        if reload_urdf:
            self._LoadRobotURDF()
            if self._on_rack:
                self.rack_constraint = (self._CreateRackConstraint(self._GetDefaultInitPosition(),
                                                                   self._GetDefaultInitOrientation()))
            self._BuildJointNameToIdDict()
            self._RemoveDefaultJointDamping()
            self._BuildMotorIdList()
            self.ResetPose(add_constraint=True)
        else:
            self._pybullet_client.resetBasePositionAndOrientation(self.quadruped,
                                                                  self._GetDefaultInitPosition(),
                                                                  self._GetDefaultInitOrientation())
            self._pybullet_client.resetBaseVelocity(self.quadruped,
                                                    [0, 0, 0],
                                                    [0, 0, 0])
            self.ResetPose(add_constraint=False)

        self._overheat_counter = np.zeros(self.num_motors)
        self._motor_enabled_list = [True] * self.num_motors
        self._observation_history.clear()
        self._step_counter = 0
        self._state_action_counter = 0
        self._is_safe = True
        self._last_action = None
        # self._SettleDownForReset(default_motor_angles, reset_time)
        if self._enable_action_filter:
            self._ResetActionFilter()

    def _LoadRobotURDF(self):
        solo12_urdf_path = self.GetURDFFile()
        if self._self_collision_enabled:
            self.quadruped = self._pybullet_client.loadURDF(
                solo12_urdf_path,
                self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation(),
                flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
        else:
            self.quadruped = self._pybullet_client.loadURDF(
                solo12_urdf_path, self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation())
        self.plane = self._pybullet_client.loadURDF("plane.urdf")

    def _SettleDownForReset(self, default_motor_angles, reset_time):
        self.ReceiveObservation()
        if reset_time <= 0:
            return

        for _ in range(500):
            self._StepInternal(
                INIT_MOTOR_ANGLES,
                motor_control_mode=robot_config.MotorControlMode.POSITION)

        if default_motor_angles is not None:
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
                self._StepInternal(
                    default_motor_angles,
                    motor_control_mode=robot_config.MotorControlMode.POSITION)

    def GetHipPositionsInBaseFrame(self):
        return _DEFAULT_HIP_POSITIONS

    def GetFootContacts(self):
        # [FR, FL, BR, BL]
        foot_ids = [2, 5, 8, 11]
        contacts = np.zeros(4)

        for leg in range(4):
            contact_points_with_ground = self.pybullet_client.getContactPoints(self.plane,
                                                                               self.quadruped,
                                                                               -1,
                                                                               foot_ids[leg])
            if len(contact_points_with_ground) > 0:
                contacts[leg] = 1

        return contacts

    def ResetPose(self, add_constraint):
        del add_constraint
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=joint_id,
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)

        for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
            # if "motor_abduction" in name:
            #     angle = INIT_MOTOR_ANGLES[i] + ABDUCTION_JOINT_OFFSET
            # elif "motor_hip" in name:
            #     angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
            # elif "motor_knee" in name:
            #     angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
            # else:
            #     raise ValueError("The name %s is not recognized as a motor joint." %
            #                      name)
            angle = INIT_MOTOR_ANGLES[i]
            self._pybullet_client.resetJointState(self.quadruped,
                                                  self._joint_name_to_id[name],
                                                  angle,
                                                  targetVelocity=0)

    def GetURDFFile(self):
        return self._urdf_filename

    def _GetMotorNames(self):
        return MOTOR_NAMES

    def _GetDefaultInitPosition(self):
        if self._on_rack:
            return INIT_RACK_POSITION
        else:
            return INIT_POSITION

    def _GetDefaultInitOrientation(self):
        # The Laikago URDF assumes the initial pose of heading towards z axis,
        # and belly towards y axis. The following transformation is to transform
        # the Laikago initial orientation to our commonly used orientation: heading
        # towards -x direction, and z axis is the up direction.
        init_orientation = pyb.getQuaternionFromEuler([0., 0., 0.])
        return init_orientation

    def GetDefaultInitPosition(self):
        """Get default initial base position."""
        return self._GetDefaultInitPosition()

    def GetDefaultInitOrientation(self):
        """Get default initial base orientation."""
        return self._GetDefaultInitOrientation()

    def GetDefaultInitJointPose(self):
        """Get default initial joint pose."""
        joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
        return joint_pose

    def ApplyAction(self, motor_commands, motor_control_mode=None):
        """Clips and then apply the motor commands using the motor model.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).N
          motor_control_mode: A MotorControlMode enum.
        """
        if self._enable_clip_motor_commands:
            motor_commands = self._ClipMotorCommands(motor_commands)
        super().ApplyAction(motor_commands, motor_control_mode)

    def _ClipMotorCommands(self, motor_commands):
        """Clips motor commands.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).

        Returns:
          Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things happens
        max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angles = self.GetMotorAngles()
        motor_commands = np.clip(motor_commands,
                                 current_motor_angles - max_angle_change,
                                 current_motor_angles + max_angle_change)
        return motor_commands

    @classmethod
    def GetConstants(cls):
        del cls
        return laikago_constants

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                                foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.

        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        assert len(self._foot_link_ids) == self.num_legs
        # toe_id = self._foot_link_ids[leg_id]

        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = list(
            range(leg_id * motors_per_leg,
                  leg_id * motors_per_leg + motors_per_leg))

        joint_angles = foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - HIP_OFFSETS[leg_id],
            l_hip_sign=(-1) ** (leg_id + 1))

        # Joint offset is necessary for Laikago.
        joint_angles = np.multiply(
            np.asarray(joint_angles) -
            np.asarray(self._motor_offset)[joint_position_idxs],
            self._motor_direction[joint_position_idxs])

        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        motor_angles = self.GetMotorAngles()
        return foot_positions_in_base_frame(motor_angles)

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles, leg_id)
