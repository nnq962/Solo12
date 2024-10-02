import numpy as np
from motion_imitation.robots import solo12_kinematic

kine = solo12_kinematic.Solo12Kinematic()

COM_OFFSET = np.array([-0.00082966, 0.00000105, -0.00060210])
HIP_OFFSETS = np.array([[0.1946, -0.0875, 0],
                        [0.1946, 0.0875, 0],
                        [-0.1946, -0.0875, 0],
                        [-0.1946, 0.0875, 0]
                        ]) + COM_OFFSET


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.16
    l_low = 0.16
    l_hip = 0.05945 * l_hip_sign
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
    l_up = 0.16
    l_low = 0.16
    l_hip = 0.05945 * l_hip_sign
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
        foot_positions[i] = kine.forward_kinematics(foot_angles[i],
                                                    l_hip_sign=(-1) ** (i + 1))
        print(foot_positions[i])
    return foot_positions + HIP_OFFSETS


angle = np.array([-0.64598868, 0.70180982, -1.43836125,
                  0.64599139, 0.70189209, -1.43836918,
                  -0.65456973, -0.70371503, 1.41936237,
                  0.65456689, -0.70365982, 1.41937767])

print(foot_positions_in_base_frame(angle).tolist())

