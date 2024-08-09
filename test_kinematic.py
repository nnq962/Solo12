import numpy as np
import matplotlib.pyplot as plt


phi = np.radians(90)
step_length = 0.08
step_height = 0.06
y_center = -0.25
no_of_points = 100
theta = 0
PI = np.pi
x_l = []
y_l = []
z_l = []

for _ in range(100):
    leg_theta = (theta / (2 * no_of_points)) * 2 * PI
    r = step_length / 2

    x = -r * np.cos(leg_theta)
    if leg_theta > PI:
        flag = 0
    else:
        flag = 1
    y = step_height * np.sin(leg_theta) * flag + y_center

    x_t, y_t, z_t = np.array(
        [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]) @ np.array(
        [x, y, 0])
    x_l.append(x_t)
    y_l.append(y_t)
    z_l.append(z_t)
    theta += 2.5
    theta = np.fmod(theta, 2 * no_of_points)


# Tạo đối tượng figure và axes 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Vẽ các điểm trên không gian 3D
ax.scatter(x_l, z_l, y_l)

# Gán nhãn cho các trục
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Hiển thị đồ thị
plt.show()
