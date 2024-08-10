import gym
import numpy as np
import pybullet
import pybullet_data
from Robot import pybullet_client
from Robot import walking_controller
from Robot import get_terrain_normal as normal_estimator
from collections import deque


def transform_action(action):
    action = np.clip(action, -1, 1)
    action[:4] = (action[:4] + 1) / 2  # Step lengths are positive always
    action[:4] = action[:4] * 2 * 0.068  # Max steplength = 2x0.068

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
                 wedge=True,
                 downhill=False,
                 deg=11,
                 imu_noise=False):

        self.incline_deg = deg
        self.render = render
        self.dt = 0.005
        self._frame_skip = 35
        self.init_position = default_pos
        self.init_orientation = default_ori
        self.no_of_points = 100
        self.frequency = 2.5
        self.theta = 0
        self.kp = 30.5
        self.kd = 0.68
        self.clips = 3
        self.on_rack = on_rack
        self.friction = 0.6
        self.gait = gait
        self.is_stairs = stairs
        self.is_wedge = wedge
        self.downhill = downhill
        self.add_imu_noise = imu_noise

        self.solo12 = None
        self._motor_id_list = None
        self._joint_name_to_id = None
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

        self.ori_history_length = 3
        self.ori_history_queue = deque([0] * 3 * self.ori_history_length,
                                       maxlen=3 * self.ori_history_length)  # observation queue

        if self.gait == 'trot':
            self.phase = [0, self.no_of_points, self.no_of_points, 0]
        elif gait == 'walk':
            self.phase = [0, self.no_of_points, 3 * self.no_of_points / 2, self.no_of_points / 2]

        if self.render:
            self.p = pybullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = pybullet_client.BulletClient()
        self.walking_controller = walking_controller.WalkingController(gait_type=self.gait, phase=self.phase)

        self.hard_reset()

        if self.is_stairs:
            boxhalflength = 0.1
            boxhalfwidth = 1
            boxhalfheight = 0.015
            sh_colbox = self.p.createCollisionShape(self.p.GEOM_BOX,
                                                    halfExtents=[boxhalflength,
                                                                 boxhalfwidth,
                                                                 boxhalfheight])
            boxorigin = 0.3
            n_steps = 15
            self.stairs = []
            for i in range(n_steps):
                step = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colbox,
                                              basePosition=[boxorigin + i * 2 * boxhalflength, 0,
                                                            boxhalfheight + i * 2 * boxhalfheight],
                                              baseOrientation=[0.0, 0.0, 0.0, 1])
                self.stairs.append(step)
                self.p.changeDynamics(step, -1, lateralFriction=0.8)

    def hard_reset(self):
        """
        Đặt các thông số mô phỏng mà sẽ duy trì không thay đổi trong suốt quá trình thử nghiệm.
        """
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(numSolverIterations=int(300))
        self.p.setGravity(0, 0, -9.81)
        self.plane = self.p.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self.p.setTimeStep(self.dt / self._frame_skip)

        if self.is_wedge:

            wedge_halfheight_offset = 0.01

            self.wedge_halfheight = wedge_halfheight_offset + 1.5 * np.tan(np.radians(self.incline_deg)) / 2.0
            self.wedgePos = [0, 0, self.wedge_halfheight]
            self.wedgeOrientation = self.p.getQuaternionFromEuler([0, 0, self.incline_ori])

            if not self.downhill:
                wedge_model_path = "Robot/Wedges/uphill/urdf/wedge_" + str(self.incline_deg) + ".urdf"

                self.init_orientation = self.p.getQuaternionFromEuler(
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

            self.wedge = self.p.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)

            self.set_wedge_friction(0.7)

        robot_path = "Robot/Simulation/solo12.urdf"
        self.solo12 = self.p.loadURDF(robot_path, self.init_position, self.init_orientation)
        self._joint_name_to_id, self._motor_id_list = self.build_motor_id_list()

        if self.on_rack:
            self.p.createConstraint(self.solo12,
                                    -1, -1, -1,
                                    self.p.JOINT_FIXED,
                                    [0, 0, 0], [0, 0, 0], [0, 0, 0.4])
        self.p.resetBasePositionAndOrientation(self.solo12, self.init_position, self.init_orientation)
        self.p.resetBaseVelocity(self.solo12, [0, 0, 0], [0, 0, 0])
        self.reset_leg()
        self.reset_abd()
        self.set_foot_friction(self.friction)

    def set_wedge_friction(self, friction):
        self.p.changeDynamics(self.wedge, -1, lateralFriction=friction)

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
        base_velocity = self.p.getBaseVelocity(self.solo12)
        return base_velocity[1]

    def get_base_linear_velocity(self):
        base_velocity = self.p.getBaseVelocity(self.solo12)
        return base_velocity[0]

    def get_motor_torques(self):
        motor_ang = [self.p.getJointState(self.solo12, motor_id)[3] for motor_id in self._motor_id_list]
        return motor_ang

    def set_foot_friction(self, foot_friction):
        foot_link_id = [2, 5, 8, 11]
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

    def get_foot_contacts(self):
        # [FR, FL, BR, BL]
        foot_ids = [2, 5, 8, 11]
        foot_contact_info = np.zeros(8)

        for leg in range(4):
            contact_points_with_ground = self.p.getContactPoints(self.plane, self.solo12, -1, foot_ids[leg])
            if len(contact_points_with_ground) > 0:
                foot_contact_info[leg] = 1

            if self.is_wedge:
                contact_points_with_wedge = self.p.getContactPoints(self.wedge, self.solo12, -1, foot_ids[leg])
                if len(contact_points_with_wedge) > 0:
                    foot_contact_info[leg + 4] = 1

            if self.is_stairs:
                for steps in self.stairs:
                    contact_points_with_stairs = self.p.getContactPoints(steps, self.solo12, -1, foot_ids[leg])
                    if len(contact_points_with_stairs) > 0:
                        foot_contact_info[leg + 4] = 1

        return foot_contact_info

    def get_observation(self):
        """
        This function returns the current observation of the environment for the interested task
        Ret:
            obs: [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t), estimated support plane (roll, pitch) ]
        """
        pos, ori = self.get_base_pos_and_orientation()
        rpy = self.p.getEulerFromQuaternion(ori)
        rpy = np.round(rpy, 5)

        for val in rpy:
            if self.add_imu_noise:
                val = add_noise(val)
            self.ori_history_queue.append(val)

        obs = np.concatenate(
            (self.ori_history_queue, [self.support_plane_estimated_roll, self.support_plane_estimated_pitch])).ravel()

        return obs

    def do_simulation(self, action, n_frames):
        omega = 2 * self.no_of_points * self.frequency
        leg_m_angle_cmd = self.walking_controller.test_elip(theta=self.theta)
        self.theta = np.fmod(omega * self.dt + self.theta, 2 * self.no_of_points)
        leg_m_angle_cmd = np.array(leg_m_angle_cmd)
        leg_m_angle_vel = np.zeros(12)

        for _ in range(n_frames):
            self.apply_pd_control(leg_m_angle_cmd, leg_m_angle_vel)
            self.p.stepSimulation()

        contact_info = self.get_foot_contacts()
        pos, ori = self.get_base_pos_and_orientation()
        rot_mat = self.p.getMatrixFromQuaternion(ori)
        rot_mat = np.array(rot_mat)
        rot_mat = np.reshape(rot_mat, (3, 3))

        (plane_normal,
         self.support_plane_estimated_roll,
         self.support_plane_estimated_pitch) = normal_estimator.vector_method_solo12(self.prev_incline_vec,
                                                                                     contact_info,
                                                                                     self.get_motor_angles(),
                                                                                     rot_mat)
        rpy_original = self.p.getEulerFromQuaternion(ori)
        # print(np.degrees(rpy_original[1]))
        # print(np.degrees(self.support_plane_estimated_pitch))
        # print("-------------------------------")

        self.prev_incline_vec = plane_normal

    def step(self, action):
        action = transform_action(action)
        self.do_simulation(action, n_frames=self._frame_skip)
        o = self.get_observation()
        reward, done = self.get_reward()
        info = {}
        return o, reward, done, info

    def get_reward(self):
        """
        Calculates reward achieved by the robot for RPY stability, torso height criterion and forward distance moved on the slope:
        Ret:
            reward : reward achieved
            done   : return True if environment terminates

        """
        wedge_angle = self.incline_deg * np.pi / 180
        robot_height_from_support_plane = 0.65
        pos, ori = self.get_base_pos_and_orientation()

        rpy_orig = self.p.getEulerFromQuaternion(ori)
        rpy = np.round(rpy_orig, 4)

        current_height = round(pos[2], 5)
        self.current_com_height = current_height
        standing_penalty = 3

        desired_height = robot_height_from_support_plane / np.cos(wedge_angle) + np.tan(wedge_angle) * (
                    (pos[0]) * np.cos(self.incline_ori) + 0.5)

        roll_reward = np.exp(-45 * ((rpy[0] - self.support_plane_estimated_roll) ** 2))
        pitch_reward = np.exp(-45 * ((rpy[1] - self.support_plane_estimated_pitch) ** 2))
        yaw_reward = np.exp(-40 * (rpy[2] ** 2))
        height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

        x = pos[0]
        y = pos[1]
        x_l = self._last_base_position[0]
        y_l = self._last_base_position[1]
        self._last_base_position = pos

        step_distance_x = (x - x_l)
        step_distance_y = abs(y - y_l)

        done = self._termination(pos, ori)
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
