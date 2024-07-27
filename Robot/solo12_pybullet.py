import gym
import numpy as np
import pybullet
import pybullet_data
from Robot import pybullet_client
from Robot import walking_controller


class Solo12PybulletEnv(gym.Env):
    """
    Solo12 Pybullet environment
    """

    def __init__(self,
                 render=True,
                 defalut_pos=(0, 0, 0.34),
                 defalut_ori=(0, 0, 0, 1),
                 on_rack=False,
                 gait="trot"):

        self.render = render
        self.dt = 0.005
        self._frame_skip = 25
        self.init_position = defalut_pos
        self.init_orientation = defalut_ori
        self.no_of_points = 100
        self.frequency = 2.5
        self.theta = 0
        self.kp = 80
        self.kd = 10
        self.clips = 3
        self.on_rack = on_rack
        self.friction = 0.6
        self.gait = gait

        self.solo12 = None
        self._motor_id_list = None
        self._joint_name_to_id = None
        self.plane = None

        if self.gait is 'trot':
            self.phase = [0, self.no_of_points, self.no_of_points, 0]
        elif gait is 'walk':
            self.phase = [0, self.no_of_points, 3 * self.no_of_points / 2, self.no_of_points / 2]

        if self.render:
            self.p = pybullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = pybullet_client.BulletClient()
        self.walking_controller = walking_controller.WalkingController(gait_type=self.gait, phase=self.phase)

        self.hard_reset()

    def hard_reset(self):
        """
        Đặt các thông số mô phỏng mà sẽ duy trì không thay đổi trong suốt quá trình thử nghiệm.
        """
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(numSolverIterations=int(300))
        self.p.setGravity(0, 0, -9.81)
        self.plane = self.p.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self.p.setTimeStep(self.dt / self._frame_skip)
        robot_path = "Robot/Simulation/solo12.urdf"
        self.solo12 = self.p.loadURDF(robot_path, self.init_position, self.init_orientation)
        self._joint_name_to_id, self._motor_id_list = self.build_motor_id_list()
        if self.on_rack:
            self.p.createConstraint(self.solo12,
                                    -1, -1, -1,
                                    self.p.JOINT_FIXED,
                                    [0, 0, 0], [0, 0, 0], [0, 0, 0.4])
        self.reset_leg()
        self.reset_abd()
        self.set_foot_friction(self.friction)

    def build_motor_id_list(self):
        num_joints = self.p.getNumJoints(self.solo12)
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self.p.getJointInfo(self.solo12, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        motor_names = ["motor_hip_fl",
                       "motor_knee_fl",
                       "motor_abduction_fl",

                       "motor_hip_hr",
                       "motor_knee_hr",
                       "motor_abduction_hr",

                       "motor_hip_fr",
                       "motor_knee_fr",
                       "motor_abduction_fr",

                       "motor_hip_hl",
                       "motor_knee_hl",
                       "motor_abduction_hl"]
        motor_id_list = [joint_name_to_id[motor_name] for motor_name in motor_names]
        return joint_name_to_id, motor_id_list

    def get_motor_angles(self):
        motor_ang = [self.p.getJointState(self.solo12, motor_id)[0] for motor_id in self._motor_id_list]
        return motor_ang

    def get_motor_velocities(self):
        motor_vel = [self.p.getJointState(self.solo12, motor_id)[1] for motor_id in self._motor_id_list]
        return motor_vel

    def get_base_pos_and_orientation(self):
        position, orientation = self.p.getBasePositionAndOrientation(self.solo12)
        return position, orientation

    def get_base_angular_velocity(self):
        basevelocity = self.p.getBaseVelocity(self.solo12)
        return basevelocity[1]

    def get_base_linear_velocity(self):
        basevelocity = self.p.getBaseVelocity(self.solo12)
        return basevelocity[0]

    def set_foot_friction(self, foot_friction):
        foot_link_id = [2, 3, 8, 11]
        for link_id in foot_link_id:
            self.p.changeDynamics(self.solo12, link_id, lateralFriction=foot_friction)
        return foot_friction

    def apply_pd_control(self, motor_commands, motor_vel_commands):
        qpos_act = self.get_motor_angles()
        qvel_act = self.get_motor_velocities()
        applied_motor_torque = self.kp * (motor_commands - qpos_act) + self.kd * (motor_vel_commands - qvel_act)
        applied_motor_torque = np.clip(np.array(applied_motor_torque), -self.clips, self.clips)
        applied_motor_torque = applied_motor_torque.tolist()

        for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
            self.set_motor_torque_by_id(motor_id, motor_torque)
        return applied_motor_torque

    def set_motor_torque_by_id(self, motor_id, torque):
        """
        Điều khiển theo momen, yêu cầu tắt điều khiển vị trí và vận tốc trước
        :param motor_id: id của động cơ
        :param torque: lực điều khiển
        :return:
        """
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=motor_id,
            controlMode=self.p.TORQUE_CONTROL,
            force=torque)

    def apply_position_control(self, target_angles):
        for motor_id, angle in zip(self._motor_id_list, target_angles):
            self.p.setJointMotorControl2(
                bodyIndex=self.solo12,
                jointIndex=motor_id,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=angle,
                force=10)

    def reset_leg(self):
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_hip_fl"],
            targetValue=-0.7, targetVelocity=0)
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_knee_fl"],
            targetValue=1.4, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_hip_fl"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_knee_fl"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)

        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_hip_fr"],
            targetValue=-0.7, targetVelocity=0)
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_knee_fr"],
            targetValue=1.4, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_hip_fr"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_knee_fr"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)

        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_hip_hl"],
            targetValue=0.7, targetVelocity=0)
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_knee_hl"],
            targetValue=-1.4, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_hip_hl"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_knee_hl"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)

        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_hip_hr"],
            targetValue=0.7, targetVelocity=0)
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_knee_hr"],
            targetValue=-1.4, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_hip_hr"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_knee_hr"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)

    def reset_abd(self):
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_abduction_fl"],
            targetValue=0,
            targetVelocity=0)
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_abduction_fr"],
            targetValue=0,
            targetVelocity=0)
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_abduction_hl"],
            targetValue=0,
            targetVelocity=0)
        self.p.resetJointState(
            self.solo12,
            self._joint_name_to_id["motor_abduction_hr"],
            targetValue=0,
            targetVelocity=0)

        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_abduction_fl"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_abduction_fr"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_abduction_hl"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)
        self.p.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=self._joint_name_to_id["motor_abduction_hr"],
            controlMode=self.p.VELOCITY_CONTROL,
            force=0, targetVelocity=0)

    def do_simulation(self, n_frames):
        omega = 2 * self.no_of_points * self.frequency
        leg_m_angle_cmd = self.walking_controller.run_elliptical(self.theta)
        self.theta = np.fmod(omega * self.dt + self.theta, 2 * self.no_of_points)
        leg_m_angle_cmd = np.array(leg_m_angle_cmd)
        leg_m_angle_vel = np.zeros(12)
        for _ in range(n_frames):
            # self.apply_position_control(leg_m_angle_cmd)
            self.apply_pd_control(leg_m_angle_cmd, leg_m_angle_vel)
            self.p.stepSimulation()

    def step(self, a=None):
        self.do_simulation(self._frame_skip)
