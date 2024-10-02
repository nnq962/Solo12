import numpy as np
import pybullet_data
from pybullet_utils import bullet_client
import pybullet as p

from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.robots import solo12_kinematic

num_motors = 12
num_legs = 4
motor_names = [
    "motor_abduction_fr", "motor_hip_fr", "motor_knee_fr",
    "motor_abduction_fl", "motor_hip_fl", "motor_knee_fl",
    "motor_abduction_hr", "motor_hip_hr", "motor_knee_hr",
    "motor_abduction_hl", "motor_hip_hl", "motor_knee_hl",
]

init_rack_position = [0, 0, 1]
init_position = [0, 0, 0.32]
joint_directions = np.ones(12)
abduction_joint_offset = 0.0
hip_joint_offset = 0.0
knee_joint_offset = 0.0
dofs_per_leg = 3

joint_offsets = np.array(
    [abduction_joint_offset, hip_joint_offset, knee_joint_offset] * 4)

max_motor_angle_change_per_step = 0.2

# update hip default positions
default_hip_positions = (
    (0.1946, -0.1015, 0),
    (0.1946, 0.1015, 0),
    (-0.1946, -0.1015, 0),
    (-0.1946, 0.1015, 0),
)

com_offset = np.array([-0.00082966, 0.00000105, -0.00060210])
hip_offsets = np.array([[0.1946, -0.0875, 0],
                        [0.1946, 0.0875, 0],
                        [-0.1946, -0.0875, 0],
                        [-0.1946, 0.0875, 0]]) + com_offset

abduction_p_gain = 100.0
abduction_d_gain = 1.
hip_p_gain = 100.0
hip_d_gain = 2.0
knee_p_gain = 100.0
knee_d_gain = 2.0

init_motor_angles = np.array([0, 0.7, -1.4,
                              0, 0.7, -1.4,
                              0, -0.7, 1.4,
                              0, -0.7, 1.4])

urdf_path = "motion_imitation/robots/simulation/solo12.urdf"


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


class Solo12(minitaur.Minitaur):
    """A simulation for the Solo12 robot."""

    MPC_BODY_MASS = 108 / 9.8
    MPC_BODY_INERTIA = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4.
    MPC_BODY_HEIGHT = 0.24
    MPC_VELOCITY_MULTIPLIER = 0.5

    def __init__(
            self,
            render=True,
            urdf_filename=urdf_path,
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
        self._render = render

        self.kinematic = solo12_kinematic.Solo12Kinematic()

        if self._render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)

        motor_kp = [
            abduction_p_gain, hip_p_gain, knee_p_gain, abduction_p_gain,
            hip_p_gain, knee_p_gain, abduction_p_gain, hip_p_gain, knee_p_gain,
            abduction_p_gain, hip_p_gain, knee_p_gain
        ]
        motor_kd = [
            abduction_d_gain, hip_d_gain, knee_d_gain, abduction_d_gain,
            hip_d_gain, knee_d_gain, abduction_d_gain, hip_d_gain, knee_d_gain,
            abduction_d_gain, hip_d_gain, knee_d_gain
        ]

        super().__init__(
            pybullet_client=self._pybullet_client,
            time_step=time_step,
            action_repeat=action_repeat,
            num_motors=num_motors,
            dofs_per_leg=dofs_per_leg,
            motor_direction=joint_directions,
            motor_offset=joint_offsets,
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
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=30)
        self._pybullet_client.setTimeStep(0.001)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self._pybullet_client.loadURDF("plane.urdf")

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
        self._SettleDownForReset(default_motor_angles, reset_time)
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

    def _SettleDownForReset(self, default_motor_angles, reset_time):
        self.ReceiveObservation()
        if reset_time <= 0:
            return

        for _ in range(500):
            self._StepInternal(init_motor_angles, motor_control_mode=robot_config.MotorControlMode.POSITION)

        if default_motor_angles is not None:
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
                self._StepInternal(default_motor_angles, motor_control_mode=robot_config.MotorControlMode.POSITION)

    def GetHipPositionsInBaseFrame(self):
        return default_hip_positions

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

        for name, i in zip(motor_names, range(len(motor_names))):
            if "motor_abduction" in name:
                angle = init_motor_angles[i] + abduction_joint_offset
            elif "motor_hip" in name:
                angle = init_motor_angles[i] + hip_joint_offset
            elif "motor_knee" in name:
                angle = init_motor_angles[i] + knee_joint_offset
            else:
                raise ValueError("The name %s is not recognized as a motor joint." %
                                 name)
            self._pybullet_client.resetJointState(self.quadruped,
                                                  self._joint_name_to_id[name],
                                                  angle,
                                                  targetVelocity=0)

    def GetURDFFile(self):
        return self._urdf_filename

    def _GetMotorNames(self):
        return motor_names

    def _GetDefaultInitPosition(self):
        if self._on_rack:
            return init_rack_position
        else:
            return init_position

    def _GetDefaultInitOrientation(self):
        return self._pybullet_client.getQuaternionFromEuler([0., 0., 0.])

    def GetDefaultInitPosition(self):
        """Get default initial base position."""
        return self._GetDefaultInitPosition()

    def GetDefaultInitOrientation(self):
        """Get default initial base orientation."""
        return self._GetDefaultInitOrientation()

    @staticmethod
    def GetDefaultInitJointPose():
        """Get default initial joint pose."""
        joint_pose = (init_motor_angles + joint_offsets) * joint_directions
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
        max_angle_change = max_motor_angle_change_per_step
        current_motor_angles = self.GetMotorAngles()
        motor_commands = np.clip(motor_commands,
                                 current_motor_angles - max_angle_change,
                                 current_motor_angles + max_angle_change)
        return motor_commands

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id, foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.

        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        # assert len(self._foot_link_ids) == self.num_legs
        # toe_id = self._foot_link_ids[leg_id]

        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = list(
            range(leg_id * motors_per_leg,
                  leg_id * motors_per_leg + motors_per_leg))

        joint_angles = self.kinematic.inverse_kinematics(foot_local_position - hip_offsets[leg_id],
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
        motor_angles = self.GetMotorAngles().reshape((4, 3))
        foot_positions = np.zeros((4, 3))
        for i in range(4):
            foot_positions[i] = self.kinematic.forward_kinematics(motor_angles[i],
                                                                  l_hip_sign=(-1) ** (i + 1))
        return foot_positions + hip_offsets

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        motor_angles = self.GetMotorAngles()[leg_id * 3:(leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles, leg_id)
