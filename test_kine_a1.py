import numpy as np

COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.],
                        [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.],
                        [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET
motor_offset = np.zeros(12)
motor_direction = np.ones(12)


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
        (2 * l_low * l_up))
    l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])


def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = np.sqrt(l_up ** 2 + l_low ** 2 +
                           2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])


def foot_positions_in_base_frame(foot_angles):
    foot_angles = foot_angles.reshape((4, 3))
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hip_frame(foot_angles[i],
                                                       l_hip_sign=(-1) ** (i + 1))
        print(foot_positions[i])
    return foot_positions + HIP_OFFSETS


angle = np.array([-0.3, 1, -1.8,
                  0.3, 1, -1.8,
                  -0.3, 1, -1.8,
                  0.3, 1, -1.8])


# print(foot_positions_in_base_frame(angle).tolist())
# [[0.14544602121832526, -0.2035496003638062, -0.21173297634595567],
# [0.14544602121832526, 0.1991776003638062, -0.21173297634595567],
# [-0.22055397878167474, -0.2035496003638062, -0.21173297634595567],
# [-0.22055397878167474, 0.1991776003638062, -0.21173297634595567]]


def ComputeMotorAnglesFromFootLocalPosition(leg_id, foot_local_position):
    """Use IK to compute the motor angles, given the foot link's local position.

    Args:
      leg_id: The leg index.
      foot_local_position: The foot link's position in the base frame.

    Returns:
      A tuple. The position indices and the angles for all joints along the
      leg. The position indices is consistent with the joint orders as returned
      by GetMotorAngles API.
    """

    motors_per_leg = 3
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = foot_position_in_hip_frame_to_joint_angle(foot_local_position - HIP_OFFSETS[leg_id],
                                                             l_hip_sign=(-1) ** (leg_id + 1))

    # Joint offset is necessary for Laikago.
    joint_angles = np.multiply(
        np.asarray(joint_angles) - np.asarray(motor_offset)[joint_position_idxs], motor_direction[joint_position_idxs])
    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles.tolist()


foot = [[0.14544602121832526, -0.2035496003638062, -0.21173297634595567],
        [0.14544602121832526, 0.1991776003638062, -0.21173297634595567],
        [-0.22055397878167474, -0.2035496003638062, -0.21173297634595567],
        [-0.22055397878167474, 0.1991776003638062, -0.21173297634595567]]

for i in range(4):
    print(ComputeMotorAnglesFromFootLocalPosition(i, foot[i]))

print(np.asarray(motor_offset)[[0, 1, 2]])
