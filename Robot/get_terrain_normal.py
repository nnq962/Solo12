import numpy as np
from Robot import solo12_kinematic
from dataclasses import dataclass
from collections import namedtuple
from collections import deque


@dataclass
class LegJointInfo:
    """A class for holding joint angles and leg label for a particular leg"""
    name: str
    hip_angle: float
    knee_angle: float
    abd_angle: float


contacts = np.zeros(4)
foot_pos_queue1 = deque([0] * 2, maxlen=2)
foot_pos_queue2 = deque([0] * 2, maxlen=2)  # observation queue


def leg_frame_to_body_frame_solo12(leg_id, hip_angle, knee_angle, abd_angle):
    """
    Function to determine the position of the foot in the body frame.
    Args:
        hip_angle : Angle of the hip joint in radians
        knee_angle: Angle of the knee joint in radians
        abd_angle : Angle of the abduction joint in radians
        leg_id    : One of the following strings: "FL", "FR", "BL", "BR"
    Ret:
        valid    : A flag to indicate if the results are valid or not
        [x, y, z]: Position co-ordinates of the foot in the body frame.

    Note: In this case the x-axis points forward, y-axis points upwards and
    the positive z-axis points towards the right.
    abd_angle is measured w.r.t the negative y-axis with CCW positive
    """
    # Robot paramters
    body_length = 0.3892
    body_width = 0.203  # 0.2939

    leg = solo12_kinematic.Solo12Kinematic()
    x_l, y_l = leg.forward_kinematics([hip_angle, knee_angle])
    valid = True

    # Return if the data is invalid
    if valid is False:
        return False, [0, 0, 0]

    y_l = y_l
    x_leg = x_l
    y_leg = y_l * np.cos(abd_angle)
    z_leg = -y_l * np.sin(abd_angle)

    # For the left legs, positive abduction angle
    # corresponds to moving outwards in the negative z direction.
    if (leg_id == "FL") | (leg_id == "BL"):
        z_leg = -1 * z_leg

    # Position of foot in leg frame
    foot_l = np.array([x_leg, y_leg, z_leg])

    if leg_id == "FL":
        leg_frame = [+body_length / 2, 0, -body_width / 2]

    elif leg_id == "FR":
        leg_frame = [+body_length / 2, 0, +body_width / 2]

    elif leg_id == "BL":
        leg_frame = [-body_length / 2, 0, -body_width / 2]

    elif leg_id == "BR":
        leg_frame = [-body_length / 2, 0, +body_width / 2]
    else:
        valid = False
        leg_frame = [0, 0, 0]

    # Position of foot in body_frame
    foot_b = foot_l + np.array(leg_frame)

    foot_b_transform = np.array([foot_b[0], -foot_b[2], foot_b[1]])

    return valid, foot_b_transform


def plane_normal_three_point(pt_a, pt_b, pt_c):
    """
    Function returns a unit vector normal to a plane
    containing the three points pt_a, pt_b, and pt_c.
    Args:
        pt_a : 3D point vector
        pt_b : 3D point vector
        pt_c : 3D point vector
    Returns:
        vec_n : 3D unit vector normal to the plane containing pt_a, pt_b, and pt_c
    """
    vec_ab = np.array(pt_b) - np.array(pt_a)
    vec_ac = np.array(pt_c) - np.array(pt_a)
    vec_n = np.cross(vec_ab, vec_ac)
    vec_n = (1 / np.linalg.norm(vec_n)) * vec_n
    return vec_n


def plane_normal_four_point(pt_a, pt_b, pt_c, pt_d):
    """
    Function returns a unit vector normal to a plane
    containing the three points pt_a, pt_b, and pt_c.
    Args:
        pt_a : 3D point vector
        pt_b : 3D point vector
        pt_c : 3D point vector
        pt_d : 3D point vector
    Returns:
        vec_n : 3D unit vector normal to the plane containing pt_a, pt_b, and pt_c
    """
    vec_ab = np.array(pt_b) - np.array(pt_a)
    vec_cd = np.array(pt_c) - np.array(pt_d)
    vec_n = np.cross(vec_ab, vec_cd)
    vec_n = (1 / np.linalg.norm(vec_n)) * vec_n
    return vec_n


def transformation(rot, x):
    """
    Represent a vector in a new frame of reference.
    The new frame differs from the original frame only
    in orientation.
    Args:
        rot : Rotation matrix
        x   : Three dimensional vector expressed in the original frame
    Returns:
        x_new : Three dimensional vector in the new frame of reference
    """
    mat = np.array(rot)
    x_in = np.array(x)
    x_out = np.dot(mat, x_in)
    return x_out


def four_point_contact_check_solo12(legs, contact_info, rot_mat):
    """
    calculates the individual vectors connecting the feet that are in contact
    by calling the forward kinematics functions.
    Args:
        legs          : An object holding the leg information and leg label of all four legs.
        contact_info  : A list containing the contact information of each individual foot
                        with the ground and a special structure (Wedge or Staircase or Track).
                        The convention being, 1 - in contact and 0 - not in contact.
        rot_mat       : The rotation matrix of the base of the robot.
    Returns:
        plane_normal                     : a np array of the caclucated plane normal
        euler_angles_of_support_plane[0] : the estimated roll of the support plane
        euler_angles_of_support_plane[1] : the estimated pitch of the support plane
    """
    vaild = False
    motor_offsets = [np.pi / 2, 0]
    leg_contact_info = np.zeros(4)
    global contacts
    for i in range(4):
        if contact_info[i] == 1 or contact_info[i + 4] == 1:
            leg_contact_info[i] = 1

    if leg_contact_info[0] == 1 and leg_contact_info[3] == 1:
        contacts[0] = 1
        contacts[3] = 1
        for i in [0, 3]:
            valid, foot_pos = leg_frame_to_body_frame_solo12(legs[i].name, legs[i].hip_angle - motor_offsets[0],
                                                             legs[i].knee_angle - motor_offsets[1], legs[i].abd_angle)
            foot_pos = transformation(rot_mat, foot_pos)
            foot_pos_queue1.append(foot_pos)

    elif leg_contact_info[1] == 1 and leg_contact_info[2] == 1:
        contacts[1] = 1
        contacts[2] = 1
        for i in [1, 2]:
            valid, foot_pos = leg_frame_to_body_frame_solo12(legs[i].name, legs[i].hip_angle - motor_offsets[0],
                                                             legs[i].knee_angle - motor_offsets[1], legs[i].abd_angle)
            foot_pos = transformation(rot_mat, foot_pos)
            foot_pos_queue2.append(foot_pos)

    if np.sum(contacts) == 4:
        vaild = True
        for i in range(4):
            contacts[i] = 0

    return vaild, foot_pos_queue2, foot_pos_queue1


def vector_method_solo12(prev_normal_vec, contact_info, motor_angles, rot_mat):
    """
    calculates the normal of the support plane, as the vector product of the
    vectors joining foots that are in contact in sucessive gait steps.
    Args:
        prev_normal_vec : The normal vector that was calculated in the previous iteration.
        contact_info  :  A list containing the contact information of each individual foot
                        with the ground and a special structure (Wedge or Staircase or Track).
                        The convention being, 1 - in contact and 0 - not in contact.
        motor_angles  : The motor angles in the order [FLH, FLK, FRH, FRK, BLH, BLK,
                        BRH, BRK, FLA, FRA, BLA, BRA]
        rot_mat       : The rotation matrix of the base of the robot.
    Returns:
        plane_normal                     : a np array of the caclucated plane normal
        euler_angles_of_support_plane[0] : the estimated roll of the support plane
        euler_angles_of_support_plane[1] : the estimated pitch of the support plane
    """

    fr = LegJointInfo("FR", motor_angles[4], motor_angles[5], motor_angles[3])
    fl = LegJointInfo("FL", motor_angles[7], motor_angles[8], motor_angles[6])
    br = LegJointInfo("BR", motor_angles[1], motor_angles[2], motor_angles[0])
    bl = LegJointInfo("BL", motor_angles[10], motor_angles[11], motor_angles[9])

    Legs = namedtuple('legs', 'front_right front_left back_right back_left')
    legs = Legs(front_right=fr, front_left=fl, back_right=br,
                back_left=bl)
    vaild, foot_contacts_vec1, foot_contacts_vec2 = four_point_contact_check_solo12(legs, contact_info, rot_mat)

    if vaild:
        normal_vec = plane_normal_four_point(foot_contacts_vec1[0], foot_contacts_vec1[1], foot_contacts_vec2[0],
                                             foot_contacts_vec2[1])
        plane_normal = normal_vec

    else:
        plane_normal = prev_normal_vec

    y_cap_of_support_plane = np.cross(plane_normal, transformation(rot_mat, np.array([1, 0, 0])))
    x_cap_of_support_plane = np.cross(y_cap_of_support_plane, plane_normal)

    # rot matrix of support plane in world frame
    rot_mat_support_plane = np.transpose(np.array([x_cap_of_support_plane, y_cap_of_support_plane, plane_normal]))

    # calculation of  euler angles of the obtained rotation matrix in world frame
    euler_angles_of_support_plane = rotation_matrix_to_euler_angles(rot_mat_support_plane)
    return np.array(plane_normal), euler_angles_of_support_plane[0], euler_angles_of_support_plane[1]


def is_rotation_matrix(r):
    """
    checks whether the given matrix satisfies the conditions of a rotation matrix
    Args:
        r : Rotation matrix to be converted
    Returns:
        A boolean value, verifying whether the given matrix is a rotation matix
    """
    rt = np.transpose(r)
    should_be_identity = np.dot(rt, r)
    i = np.identity(3, dtype=r.dtype)
    n = np.linalg.norm(i - should_be_identity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(r):
    """
    Coverts rotation matrix to euler angles
    Args:
        r : Rotation matrix to be converted
    Returns:
        [x,y,z] : The list of euler angles in the order roll(x), pitch(y) and yaw(z)

    """
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        y = np.arctan2(-r[2, 0], sy)
        z = 0

    return np.array([x, y, z])
