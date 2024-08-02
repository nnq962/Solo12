from dataclasses import dataclass
from collections import namedtuple
from Robot import solo12_kinematic
import numpy as np


@dataclass
class LegData:
    name: str
    motor_hip: float = 0.0
    motor_knee: float = 0.0
    motor_abduction: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    b: float = 1.0
    step_length: float = 0.0
    x_shift = 0.0
    y_shift = 0.0
    z_shift = 0.0


@dataclass
class RobotData:
    front_right: float
    front_left: float
    back_right: float
    back_left: float


class WalkingController:
    def __init__(self,
                 gait_type='trot',
                 phase=(0, 0, 0, 0),
                 no_of_points=100):
        self._phase = RobotData(front_right=phase[0], front_left=phase[1], back_right=phase[2], back_left=phase[3])
        self.front_left = LegData('fl')
        self.front_right = LegData('fr')
        self.back_left = LegData('bl')
        self.back_right = LegData('br')
        self.gait_type = gait_type
        self.no_of_points = no_of_points

        self.motor_offsets = [np.pi / 2, np.radians(0)]
        self.leg_name_to_sol_branch_Solo12 = {'fl': 1, 'fr': 1, 'bl': 0, 'br': 0}
        self.Solo12_Kin = solo12_kinematic.Solo12Kinematic()

    def update_leg_theta(self, theta):
        self.front_right.theta = np.fmod(theta + self._phase.front_right, 2 * self.no_of_points)
        self.front_left.theta = np.fmod(theta + self._phase.front_left, 2 * self.no_of_points)
        self.back_right.theta = np.fmod(theta + self._phase.back_right, 2 * self.no_of_points)
        self.back_left.theta = np.fmod(theta + self._phase.back_left, 2 * self.no_of_points)

    def initialize_leg_state(self, theta):
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right=self.front_right, front_left=self.front_left, back_right=self.back_right,
                    back_left=self.back_left)
        self.update_leg_theta(theta)
        return legs

    def forward(self, step_length, theta):
        legs = self.initialize_leg_state(theta)

        # Parameters for elip --------------------
        step_length = 0.0
        step_height = 0.0
        x_center = 0.
        y_center = -0.25
        z_center = 0.
        # ----------------------------------------

        x = y = 0
        for leg in legs:
            leg_theta = (leg.theta / (2 * self.no_of_points)) * 2 * np.pi
            leg.r = step_length / 2
            if self.gait_type == "trot":
                if leg.name == "fl" or leg.name == "fr":
                    x = -leg.r * np.cos(leg_theta) + x_center
                else:
                    x = -leg.r * np.cos(leg_theta) - x_center
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = step_height * np.sin(leg_theta) * flag + y_center

            leg.x, leg.y, leg.z = x, y, 0

            (leg.motor_knee,
             leg.motor_hip,
             leg.motor_abduction) = self.Solo12_Kin.inverse_kinematics(leg.x,
                                                                       leg.y,
                                                                       leg.z,
                                                                       self.leg_name_to_sol_branch_Solo12[leg.name])

            leg.motor_hip = leg.motor_hip + self.motor_offsets[0]
            leg.motor_knee = leg.motor_knee + self.motor_offsets[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_left.motor_abduction,
                            legs.back_right.motor_hip, legs.back_right.motor_knee, legs.back_right.motor_abduction,
                            legs.front_right.motor_hip, legs.front_right.motor_knee, legs.front_right.motor_abduction,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_left.motor_abduction]

        return leg_motor_angles

    def sidesteps(self, theta):
        legs = self.initialize_leg_state(theta)

        # Parameters for elip --------------------
        step_length = 0.02
        step_height = 0.04
        x_center = 0.
        y_center = -0.25
        # ----------------------------------------

        x = y = 0
        for leg in legs:
            leg_theta = (leg.theta / (2 * self.no_of_points)) * 2 * np.pi
            leg.r = step_length / 2
            if self.gait_type == "trot":
                x = -leg.r * np.cos(leg_theta) - x_center
                if leg_theta > np.pi:
                    flag = 0
                else:
                    flag = 1
                y = step_height * np.sin(leg_theta) * flag + y_center
            leg.x, leg.y, leg.z = 0, y, x

            (leg.motor_knee,
             leg.motor_hip,
             leg.motor_abduction) = self.Solo12_Kin.inverse_kinematics(leg.x,
                                                                       leg.y,
                                                                       leg.z,
                                                                       self.leg_name_to_sol_branch_Solo12[leg.name])

            leg.motor_hip = leg.motor_hip + self.motor_offsets[0]
            leg.motor_knee = leg.motor_knee + self.motor_offsets[1]

        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_left.motor_abduction,
                            legs.back_right.motor_hip, legs.back_right.motor_knee, legs.back_right.motor_abduction,
                            legs.front_right.motor_hip, legs.front_right.motor_knee, legs.front_right.motor_abduction,
                            legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_left.motor_abduction]

        return leg_motor_angles
