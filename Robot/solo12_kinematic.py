import numpy as np


class Solo12Kinematic:
    """
    Solo12 kinematic model
    """
    def __init__(self, l1=0.16, l2=0.16):
        self.l1 = l1
        self.l2 = l2

    def inverse_2_d(self, x, y, br):
        sol_branch = br
        t1 = ((-4 * self.l2 * y + np.sqrt(
            16 * self.l2 ** 2 * y ** 2 - 4 * (-self.l1 ** 2 + self.l2 ** 2 - 2 * self.l2 * x + x ** 2 + y ** 2)
            * (-self.l1 ** 2 + self.l2 ** 2 + 2 * self.l2 * x + x ** 2 + y ** 2)))
              / (2. * (self.l1 ** 2 - self.l2 ** 2 - 2 * self.l2 * x - x ** 2 - y ** 2)))

        t2 = (-4 * self.l2 * y - np.sqrt(
            16 * self.l2 ** 2 * y ** 2 - 4 * (-self.l1 ** 2 + self.l2 ** 2 - 2 * self.l2 * x + x ** 2 + y ** 2)
            * (-self.l1 ** 2 + self.l2 ** 2 + 2 * self.l2 * x + x ** 2 + y ** 2))) / (
                         2. * (self.l1 ** 2 - self.l2 ** 2 - 2 * self.l2 * x - x ** 2 - y ** 2))

        if sol_branch:
            t = t2
        else:
            t = t1
        th12 = np.arctan2(2 * t, (1 - t ** 2))
        th1 = np.arctan2(y - self.l2 * np.sin(th12), x - self.l2 * np.cos(th12))
        th2 = th12 - th1
        return [th1, th2]

    def inverse_kinematics(self, x, y, z, br):
        theta = np.arctan2(z, -y)
        new_coords = np.array([x, y / np.cos(theta), z])
        motor_hip, motor_knee = self.inverse_2_d(new_coords[0], new_coords[1], br)
        return motor_knee, motor_hip, theta

    def forward_kinematics(self, q):
        """
        Forward kinematics of the Solo12
        Args:
        -- q : Active joint angles, i.e., [theta1, theta4], angles of the links 1 and 4 (the driven links)
        Return:
        -- valid : Specifies if the result is valid
        -- x : End-effector position
        """
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        return [x, y]
