import numpy as np


def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.16
    l_low = 0.16
    l_hip = 0.0 * l_hip_sign
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
    l_hip = 0.0 * l_hip_sign
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


f_p = [0.03019163492206934, 0.08505, -0.3450919662651284]
ang = np.radians([0, -25.8719, 74.3636])
# print(ang)
position = foot_position_in_hip_frame(ang)
print("angles   :", np.degrees(foot_position_in_hip_frame_to_joint_angle(position)))
print("xyz      :", foot_position_in_hip_frame(ang, 1))

from motion_imitation.robots import solo12_kinematic
obj = solo12_kinematic.Solo12Kinematic()
print(np.degrees(obj.inverse_kinematics(0.05, -0.25, 0, 1)))
