import gym
from gym import spaces
import numpy as np
import pybullet
import pybullet_data
from Robot import pybullet_client
from Robot import walking_controller
from Robot import get_terrain_normal as normal_estimator
from collections import deque

motor_names = [
    "motor_hip_fl", "motor_knee_fl", "motor_abduction_fl",
    "motor_hip_hr", "motor_knee_hr", "motor_abduction_hr",
    "motor_hip_fr", "motor_knee_fr", "motor_abduction_fr",
    "motor_hip_hl", "motor_knee_hl", "motor_abduction_hl"
]


def transform_action(action):
    action = np.clip(action, -1, 1)
    action[:4] = (action[:4] + 1) / 2  # Step lengths are positive always
    action[:4] = action[:4] * 0.136  # Max steplength = 0.136
    action[4:8] = action[4:8] * np.pi / 2  # PHI can be [-pi/2, pi/2]
    action[8:12] = 0.07 * (action[8:12] + 1) / 2  # elipse center y is positive always
    action[12:16] = action[12:16] * 0.04  # x
    action[16:20] = action[16:20] * 0.035  # Max allowed Z-shift due to abduction limits is 3.5cm
    action[17] = -action[17]
    action[19] = -action[19]

    return action


def add_noise(sensor_value, sd=0.04):
    """
    Adds sensor noise of user defined standard deviation in current sensor_value
    """
    noise = np.random.normal(0, sd, 1)
    sensor_value = sensor_value + noise[0]
    return sensor_value


class Solo12PybulletEnv(gym.Env):
    """
    Solo12 Pybullet environment
    """

    def __init__(self,
                 render=True,
                 default_pos=(0, 0, 0.33),
                 default_ori=(0, 0, 0, 1),
                 on_rack=False,
                 gait="trot",
                 stairs=False,
                 wedge=False,
                 downhill=False,
                 deg=11,
                 imu_noise=False,
                 end_steps=1000,
                 pd_control_enabled=True,
                 motor_kp=30.5,
                 motor_kd=0.68,
                 extension_amplitude=0.08,
                 swing_amplitude=0.06):

        self.pd_control_enabled = pd_control_enabled
        self.incline_deg = deg
        self.render = render
        self.dt = 0.005
        self.frame_skip = 25
        self.init_position = default_pos
        self.init_orientation = default_ori
        self.no_of_points = 100
        self.frequency = 2.5
        self.theta = 0
        self.kp = motor_kp  # 30.5
        self.kd = motor_kd  # 0.68
        self.clips = 3
        self.on_rack = on_rack
        self.friction = 0.6
        self.gait = gait
        self.is_stairs = stairs
        self.is_wedge = wedge
        self.downhill = downhill
        self.add_imu_noise = imu_noise
        self.termination_steps = end_steps
        self.extension_amplitude = extension_amplitude
        self.swing_amplitude = swing_amplitude

        self.solo12 = None
        self.motor_id_list = None
        self.joint_name_to_id = None
        self.plane = None
        self.wedge_halfheight = None
        self.wedgePos = None
        self.wedgeOrientation = None
        self.robot_landing_height = None
        self.wedge = None
        self.support_plane_estimated_pitch = 0
        self.support_plane_estimated_roll = 0
        self.incline_ori = 0
        self.wedge_start = 0.5
        self.prev_incline_vec = (0, 0, 1)
        self.last_base_position = [0, 0, 0]
        self.n_steps = 0

        self.ori_history_length = 3
        self.ori_history_queue = deque([0] * 3 * self.ori_history_length,
                                       maxlen=3 * self.ori_history_length)  # observation queue

        if self.gait == 'trot':
            self.phase = [0, self.no_of_points, self.no_of_points, 0]
        elif gait == 'walk':
            self.phase = [0, self.no_of_points, 3 * self.no_of_points / 2, self.no_of_points / 2]

        if self.render:
            self.pybullet_client = pybullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.pybullet_client = pybullet_client.BulletClient()
        self.walking_controller = walking_controller.WalkingController(gait_type=self.gait, phase=self.phase)

        self.hard_reset()

        if self.is_stairs:
            boxhalflength = 0.1
            boxhalfwidth = 1
            boxhalfheight = 0.015
            sh_colbox = self.pybullet_client.createCollisionShape(self.pybullet_client.GEOM_BOX,
                                                                  halfExtents=[boxhalflength,
                                                                               boxhalfwidth,
                                                                               boxhalfheight])
            boxorigin = 0.3
            n_steps = 15
            self.stairs = []
            for i in range(n_steps):
                step = self.pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colbox,
                                                            basePosition=[boxorigin + i * 2 * boxhalflength, 0,
                                                                          boxhalfheight + i * 2 * boxhalfheight],
                                                            baseOrientation=[0.0, 0.0, 0.0, 1])
                self.stairs.append(step)
                self.pybullet_client.changeDynamics(step, -1, lateralFriction=0.8)

        abduction_low = np.radians(-45)
        abduction_high = np.radians(45)
        other_motor_low = np.radians(-90)
        other_motor_high = np.radians(90)

        action_low = np.array([other_motor_low, other_motor_low, abduction_low,
                               other_motor_low, other_motor_low, abduction_low,
                               other_motor_low, other_motor_low, abduction_low,
                               other_motor_low, other_motor_low, abduction_low], dtype=np.float32)

        action_high = np.array([other_motor_high, other_motor_high, abduction_high,
                                other_motor_high, other_motor_high, abduction_high,
                                other_motor_high, other_motor_high, abduction_high,
                                other_motor_high, other_motor_high, abduction_high], dtype=np.float32)

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        observation_dim = len(self.get_observation())
        observation_low = -np.inf * np.ones(observation_dim, dtype=np.float32)
        observation_high = np.inf * np.ones(observation_dim, dtype=np.float32)

        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

    def hard_reset(self):
        """
        Đặt các thông số mô phỏng mà sẽ duy trì không thay đổi trong suốt quá trình thử nghiệm.
        """
        self.pybullet_client.resetSimulation()
        self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self.pybullet_client.setGravity(0, 0, -9.81)
        self.plane = self.pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self.pybullet_client.setTimeStep(self.dt / self.frame_skip)

        if self.is_wedge:

            wedge_halfheight_offset = 0.01

            self.wedge_halfheight = wedge_halfheight_offset + 1.5 * np.tan(np.radians(self.incline_deg)) / 2.0
            self.wedgePos = [0, 0, self.wedge_halfheight]
            self.wedgeOrientation = self.pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

            if not self.downhill:
                wedge_model_path = "Robot/Wedges/uphill/urdf/wedge_" + str(self.incline_deg) + ".urdf"

                self.init_orientation = self.pybullet_client.getQuaternionFromEuler(
                    [np.radians(self.incline_deg) * np.sin(self.incline_ori),
                     -np.radians(self.incline_deg) * np.cos(self.incline_ori), 0])

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * abs(self.wedge_start)

                self.init_position = [self.init_position[0], self.init_position[1], self.robot_landing_height]

            else:
                wedge_model_path = "Robot/Wedges/downhill/urdf/wedge_" + str(
                    self.incline_deg) + ".urdf"

                self.robot_landing_height = wedge_halfheight_offset + 0.28 + np.tan(
                    np.radians(self.incline_deg)) * 1.5

                self.init_position = [0, 0, self.robot_landing_height]  # [0.5, 0.7, 0.3] #[-0.5,-0.5,0.3]

                self.init_orientation = [0, 0, 0, 1]

            self.wedge = self.pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)

            self.set_wedge_friction(0.7)

        robot_path = "Robot/Simulation/solo12.urdf"
        self.solo12 = self.pybullet_client.loadURDF(robot_path, self.init_position, self.init_orientation)
        self.joint_name_to_id, self.motor_id_list = self.build_motor_id_list()

        if self.on_rack:
            self.pybullet_client.createConstraint(self.solo12,
                                                  -1, -1, -1,
                                                  self.pybullet_client.JOINT_FIXED,
                                                  [0, 0, 0], [0, 0, 0], [0, 0, 0.4])
        self.pybullet_client.resetBasePositionAndOrientation(self.solo12, self.init_position, self.init_orientation)
        self.pybullet_client.resetBaseVelocity(self.solo12, [0, 0, 0], [0, 0, 0])
        self.reset_pose()
        self.set_foot_friction(self.friction)

    def reset(self, **kwargs):
        self.theta = 0
        self.last_base_position = [0, 0, 0]

        self.pybullet_client.resetBasePositionAndOrientation(self.solo12, self.init_position, self.init_orientation)
        self.pybullet_client.resetBaseVelocity(self.solo12, [0, 0, 0], [0, 0, 0])
        self.reset_pose()

        self.n_steps = 0
        return self.get_observation()

    def set_wedge_friction(self, friction):
        self.pybullet_client.changeDynamics(self.wedge, -1, lateralFriction=friction)

    def build_motor_id_list(self):
        num_joints = self.pybullet_client.getNumJoints(self.solo12)
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.solo12, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        motor_id_list = [joint_name_to_id[motor_name] for motor_name in motor_names]
        return joint_name_to_id, motor_id_list

    def get_motor_angles(self):
        motor_ang = [self.pybullet_client.getJointState(self.solo12, motor_id)[0] for motor_id in self.motor_id_list]
        return motor_ang

    def get_motor_velocities(self):
        motor_vel = [self.pybullet_client.getJointState(self.solo12, motor_id)[1] for motor_id in self.motor_id_list]
        return motor_vel

    def get_base_pos_and_orientation(self):
        position, orientation = self.pybullet_client.getBasePositionAndOrientation(self.solo12)
        return position, orientation

    def get_base_angular_velocity(self):
        base_velocity = self.pybullet_client.getBaseVelocity(self.solo12)
        return base_velocity[1]

    def get_base_linear_velocity(self):
        base_velocity = self.pybullet_client.getBaseVelocity(self.solo12)
        return base_velocity[0]

    def get_motor_torques(self):
        motor_ang = [self.pybullet_client.getJointState(self.solo12, motor_id)[3] for motor_id in self.motor_id_list]
        return motor_ang

    def set_foot_friction(self, foot_friction):
        foot_link_id = [2, 5, 8, 11]
        for link_id in foot_link_id:
            self.pybullet_client.changeDynamics(self.solo12, link_id, lateralFriction=foot_friction)
        return foot_friction

    def apply_pd_control(self, motor_commands):
        motor_vel_commands = np.zeros(12)
        qpos_act = self.get_motor_angles()
        qvel_act = self.get_motor_velocities()
        applied_motor_torque = self.kp * (motor_commands - qpos_act) + self.kd * (motor_vel_commands - qvel_act)
        applied_motor_torque = np.clip(np.array(applied_motor_torque), -self.clips, self.clips)
        applied_motor_torque = applied_motor_torque.tolist()

        for motor_id, motor_torque in zip(self.motor_id_list, applied_motor_torque):
            self.set_motor_torque_by_id(motor_id, motor_torque)
        return applied_motor_torque

    def apply_postion_control(self, desired_angles):
        for motor_id, angle in zip(self.motor_id_list, desired_angles):
            self.set_desired_motor_angle_by_id(motor_id, angle)

    def set_motor_torque_by_id(self, motor_id, torque):
        self.pybullet_client.setJointMotorControl2(
            bodyIndex=self.solo12,
            jointIndex=motor_id,
            controlMode=self.pybullet_client.TORQUE_CONTROL,
            force=torque)

    def set_desired_motor_angle_by_id(self, motor_id, desired_angle):
        self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                   jointIndex=motor_id,
                                                   controlMode=self.pybullet_client.POSITION_CONTROL,
                                                   targetPosition=desired_angle,
                                                   positionGain=1,
                                                   velocityGain=1,
                                                   force=3)

    def set_desired_motor_angle_by_name(self, motor_name, desired_angle):
        self.set_desired_motor_angle_by_id(self.joint_name_to_id[motor_name], desired_angle)

    def reset_leg(self):
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_hip_fl"],
                                             targetValue=-0.7, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_knee_fl"],
                                             targetValue=1.4, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_hip_fr"],
                                             targetValue=-0.7, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_knee_fr"],
                                             targetValue=1.4, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_hip_hl"],
                                             targetValue=0.7, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_knee_hl"],
                                             targetValue=-1.4, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_hip_hr"],
                                             targetValue=0.7, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_knee_hr"],
                                             targetValue=-1.4, targetVelocity=0)

        if self.pd_control_enabled:
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_hip_fl"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_knee_fl"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_hip_fr"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_knee_fr"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_hip_hl"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_knee_hl"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_hip_hr"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_knee_hr"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
        else:
            self.set_desired_motor_angle_by_name("motor_hip_fl", desired_angle=-0.7)
            self.set_desired_motor_angle_by_name("motor_knee_fl", desired_angle=1.4)

            self.set_desired_motor_angle_by_name("motor_hip_fr", desired_angle=-0.7)
            self.set_desired_motor_angle_by_name("motor_knee_fr", desired_angle=1.4)

            self.set_desired_motor_angle_by_name("motor_hip_hl", desired_angle=0.7)
            self.set_desired_motor_angle_by_name("motor_knee_hl", desired_angle=-1.4)

            self.set_desired_motor_angle_by_name("motor_hip_hr", desired_angle=0.7)
            self.set_desired_motor_angle_by_name("motor_knee_hr", desired_angle=-1.4)

    def reset_abd(self):
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_abduction_fl"],
                                             targetValue=0, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_abduction_fr"],
                                             targetValue=0, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_abduction_hl"],
                                             targetValue=0, targetVelocity=0)
        self.pybullet_client.resetJointState(self.solo12,
                                             self.joint_name_to_id["motor_abduction_hr"],
                                             targetValue=0, targetVelocity=0)
        if self.pd_control_enabled:
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_abduction_fl"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_abduction_fr"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_abduction_hl"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
            self.pybullet_client.setJointMotorControl2(bodyIndex=self.solo12,
                                                       jointIndex=self.joint_name_to_id["motor_abduction_hr"],
                                                       controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                       force=0, targetVelocity=0)
        else:
            self.set_desired_motor_angle_by_name("motor_abduction_fl", desired_angle=0)
            self.set_desired_motor_angle_by_name("motor_abduction_fr", desired_angle=0)
            self.set_desired_motor_angle_by_name("motor_abduction_hl", desired_angle=0)
            self.set_desired_motor_angle_by_name("motor_abduction_hr", desired_angle=0)

    def reset_pose(self):
        self.reset_abd()
        self.reset_leg()

    def get_foot_contacts(self):
        # [FR, FL, BR, BL]
        foot_ids = [2, 5, 8, 11]
        foot_contact_info = np.zeros(8)

        for leg in range(4):
            contact_points_with_ground = self.pybullet_client.getContactPoints(self.plane, self.solo12, -1,
                                                                               foot_ids[leg])
            if len(contact_points_with_ground) > 0:
                foot_contact_info[leg] = 1

            if self.is_wedge:
                contact_points_with_wedge = self.pybullet_client.getContactPoints(self.wedge, self.solo12, -1,
                                                                                  foot_ids[leg])
                if len(contact_points_with_wedge) > 0:
                    foot_contact_info[leg + 4] = 1

            if self.is_stairs:
                for steps in self.stairs:
                    contact_points_with_stairs = self.pybullet_client.getContactPoints(steps, self.solo12, -1,
                                                                                       foot_ids[leg])
                    if len(contact_points_with_stairs) > 0:
                        foot_contact_info[leg + 4] = 1

        return foot_contact_info

    # def get_observation(self):
    #     """
    #     This function returns the current observation of the environment for the interested task
    #     Ret:
    #         obs: [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t),
    #          estimated support plane (roll, pitch)]
    #     """
    #     pos, ori = self.get_base_pos_and_orientation()
    #     rpy = self.pybullet_client.getEulerFromQuaternion(ori)
    #     rpy = np.round(rpy, 5)
    #
    #     for val in rpy:
    #         if self.add_imu_noise:
    #             val = add_noise(val)
    #         self.ori_history_queue.append(val)
    #
    #     obs = np.concatenate((self.ori_history_queue,
    #                           [self.support_plane_estimated_roll, self.support_plane_estimated_pitch])).ravel()
    #
    #     return obs

    def get_observation(self):
        motor_angles = np.array(self.get_motor_angles(), dtype=np.float32)
        motor_velocities = np.array(self.get_motor_velocities(), dtype=np.float32)

        _, ori = self.get_base_pos_and_orientation()
        rpy = self.pybullet_client.getEulerFromQuaternion(ori)
        rpy = np.array(rpy, dtype=np.float32)  # Radians

        observation = np.concatenate((motor_angles, motor_velocities, rpy))

        return observation

    def estimate_terrain(self):
        contact_info = self.get_foot_contacts()
        pos, ori = self.get_base_pos_and_orientation()
        rot_mat = self.pybullet_client.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat)
        rot_mat = np.reshape(rot_mat, (3, 3))

        (plane_normal,
         self.support_plane_estimated_roll,
         self.support_plane_estimated_pitch) = normal_estimator.vector_method_solo12(self.prev_incline_vec,
                                                                                     contact_info,
                                                                                     self.get_motor_angles(),
                                                                                     rot_mat)
        self.prev_incline_vec = plane_normal

    def apply_action(self, action):
        action = None
        # Todo: change action
        motor_commands = self.walking_controller.test_elip(theta=self.theta)
        motor_commands = np.array(motor_commands)

        # Update theta
        omega = 2 * self.no_of_points * self.frequency
        self.theta = np.fmod(omega * self.dt + self.theta, 2 * self.no_of_points)

        force_visualizing_counter = 0
        # action = np.array(action)

        # Apply action
        for _ in range(self.frame_skip):
            if self.pd_control_enabled:
                self.apply_pd_control(motor_commands)
            else:
                self.apply_postion_control(motor_commands)
            self.pybullet_client.stepSimulation()
            if self.n_steps % 300 == 0:
                force_visualizing_counter += 1
                link = np.random.randint(0, 11)
                pertub_range = [0, -1200, 1200, -2000, 2000]
                y_force = pertub_range[np.random.randint(0, 4)]
                if force_visualizing_counter % 10 == 0:
                    self.apply_ext_force(x_f=0, y_f=y_force, link_index=1, visualize=True, life_time=3)

        self.n_steps += 1

    def step(self, action):
        self.apply_action(action)
        self.estimate_terrain()
        ob = self.get_observation()
        reward, done = self.reward()
        info = {}
        return ob, reward, done, info

    def termination(self, pos, orientation):
        """
        Check termination conditions of the environment
        Args:
            pos 		: current position of the robot's base in world frame
            orientation : current orientation of robot's base (Quaternions) in world frame
        Ret:
            done 		: return True if termination conditions satisfied
        """
        done = False
        rpy = self.pybullet_client.getEulerFromQuaternion(orientation)

        if self.n_steps >= self.termination_steps:
            done = True
        else:
            if abs(rpy[0]) > np.radians(30):
                print('Oops, Robot about to fall sideways! Terminated')
                done = True

            if abs(rpy[1]) > np.radians(35):
                print('Oops, Robot doing wheely! Terminated')
                done = True

            if pos[2] > 0.9:
                print('Robot was too high! Terminated')
                done = True

        return done

    def get_reward(self):
        """
        Calculates reward achieved by the robot for RPY stability, torso height criterion
         and forward distance moved on the slope:
        Ret:
            reward : reward achieved
            done   : return True if environment terminates

        """
        wedge_angle = self.incline_deg * np.pi / 180
        robot_height_from_support_plane = 0.65
        pos, ori = self.get_base_pos_and_orientation()

        rpy_orig = self.pybullet_client.getEulerFromQuaternion(ori)
        rpy = np.round(rpy_orig, 4)

        current_height = round(pos[2], 5)
        # standing_penalty = 3

        desired_height = robot_height_from_support_plane / np.cos(wedge_angle) + np.tan(wedge_angle) * (
                (pos[0]) * np.cos(self.incline_ori) + 0.5)

        roll_reward = np.exp(-45 * ((rpy[0] - self.support_plane_estimated_roll) ** 2))
        pitch_reward = np.exp(-45 * ((rpy[1] - self.support_plane_estimated_pitch) ** 2))
        yaw_reward = np.exp(-40 * (rpy[2] ** 2))
        height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

        x = pos[0]
        y = pos[1]
        x_l = self.last_base_position[0]
        y_l = self.last_base_position[1]
        self.last_base_position = pos

        step_distance_x = (x - x_l)
        step_distance_y = abs(y - y_l)

        done = self.termination(pos, ori)

        if done:
            reward = 0
        else:
            reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4) \
                     + round(height_reward, 4) + 100 * round(step_distance_x, 4) - 20 * round(step_distance_y, 4)

            '''
            #Penalize for standing at same position for continuous 150 steps
            self.step_disp.append(step_distance_x)

            if(self._n_steps>150):
                if(sum(self.step_disp)<0.035):
                    reward = reward-standing_penalty
            '''

        return reward, done

    def reward(self):
        pos, ori = self.get_base_pos_and_orientation()
        rpy_orig = self.pybullet_client.getEulerFromQuaternion(ori)
        rpy = np.round(rpy_orig, 4)
        x_reward = pos[0] - self.last_base_position[0]
        y_reward = -np.abs(pos[1])
        roll_reward = -np.abs(np.degrees(rpy[0]))
        pitch_reward = -np.abs(np.degrees(rpy[1]))
        yaw_reward = -np.abs(np.degrees(rpy[2]))

        done = self.termination(pos, ori)

        if done:
            reward = -20
        else:
            reward = 2 * x_reward + y_reward + roll_reward + pitch_reward + yaw_reward

        return reward, done

    def apply_ext_force(self, x_f, y_f, link_index=1, visualize=False, life_time=0.01):
        """
        Function to apply external force on the robot
        :param x_f: external force in x direction
        :param y_f: external force in y direction
        :param link_index: link index of the robot where the force needs to be applied
        :param visualize: bool, whether to visualize external force by arrow symbols
        :param life_time: lifetime of the visualization
        :return:
        """
        force_applied = [x_f, y_f, 0]
        self.pybullet_client.applyExternalForce(self.solo12, link_index, forceObj=force_applied, posObj=[0, 0, 0],
                                                flags=self.pybullet_client.LINK_FRAME)
        f_mag = np.linalg.norm(np.array(force_applied))

        if visualize and f_mag != 0.0:
            point_of_force = self.pybullet_client.getLinkState(self.solo12, link_index)[0]

            lam = 1 / (2 * f_mag)
            dummy_pt = [point_of_force[0] - lam * force_applied[0],
                        point_of_force[1] - lam * force_applied[1],
                        point_of_force[2] - lam * force_applied[2]]
            self.pybullet_client.addUserDebugText(str(round(f_mag, 2)) + " N", dummy_pt, [0.13, 0.54, 0.13],
                                                  textSize=2, lifeTime=life_time)
            self.pybullet_client.addUserDebugLine(point_of_force, dummy_pt, [0, 0, 1], 3, lifeTime=life_time)

    def gen_signal(self, t, phase):
        """Generates a sinusoidal reference leg trajectory.

        The foot (leg tip) will move in a ellipse specified by extension and swing
        amplitude.

        Args:
          t: Current time in simulation.
          phase: The phase offset for the periodic trajectory.

        Returns:
          The desired leg extension and swing angle at the current time.
        """
        period = 1 / self.frequency
        extension = self.extension_amplitude * np.cos(2 * np.pi / period * t + phase)
        swing = self.swing_amplitude * np.sin(2 * np.pi / period * t + phase)
        return extension, swing

    def signal(self, t):
        """Generates the trotting gait for the robot.

        Args:
          t: Current time in simulation.

        Returns:
          A numpy array of the reference leg positions.
        """
        # Generates the leg trajectories for the two digonal pair of legs.
        ext_first_pair, sw_first_pair = self.gen_signal(t, 0)
        ext_second_pair, sw_second_pair = self.gen_signal(t, np.pi)

        trotting_signal = np.array([
            sw_first_pair, sw_second_pair, sw_second_pair, sw_first_pair, ext_first_pair,
            ext_second_pair, ext_second_pair, ext_first_pair
        ])
        signal = np.array(self.init_pose) + trotting_signal
        return signal
