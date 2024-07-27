import matplotlib.pyplot as plt
import numpy as np
from Robot import solo12_kinematic_2

sol12 = solo12_kinematic_2.Solo12Kinematic()
base1 = sol12.base_pivot1
[l1, l2, l3, l4] = sol12.link_parameters
pi = np.pi

motor_hip = []
motor_knee = []


def draw(p1, p2):
    _, [q1, q2] = sol12.inverse2d(ee_pos=[p1, p2])
    print(np.degrees(q1))
    print(np.degrees(q2))

    # if q1 > 0:
    #     print(-2 * np.pi + q1)
    #     print("------------")
    # print(q1)
    # print(q2)
    # print("------------")

    x_00, y_00 = 0, 0

    x1, y1 = l1 * np.cos(q1), l1 * np.sin(q1)
    x2, y2 = x1 + l2 * np.cos(q2), y1 + l2 * np.sin(q2)
    x_end, y_end = p1, p2
    plt.plot([x_00, x1], [y_00, y1], 'ro-', label='l1')
    plt.plot([x1, x2], [y1, y2], 'go-', label='l2')
    plt.plot(x, y, color='blue', label='Quỹ đạo')

    plt.title('Leg tracjectory')
    plt.xlabel('X-axis (m)')
    plt.ylabel('Y-axis (m)')
    plt.xlim(-0.15, 0.2)
    plt.ylim(-0.35, 0.01)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    # plt.pause(0.0000001)
    plt.clf()
    plt.show()


no_of_points = 100
step_length = 0.1
radius = step_length / 2
theta = 0
y_center = -0.26
foot_clearance = 0.05
x_shift = 0.01
y_shift = 0
count = 5

x = []
y = []

# for _ in range(count * 80):
#     leg_theta = (theta / (2 * no_of_points)) * 2 * np.pi
#
#     t = -radius * np.cos(leg_theta) + x_shift
#     x.append(t)
#     if leg_theta > np.pi:
#         flag = 0
#     else:
#         flag = 1
#     z = foot_clearance * np.sin(leg_theta) * flag + y_center + y_shift
#     y.append(z)
#
#     theta += 2.5
#     theta = np.fmod(theta, 2 * no_of_points)
#     draw(t, z)

draw(0, -0.25)
