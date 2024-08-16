from Robot import solo12_pybullet
from Robot import solo12_kinematic
import numpy as np


# Hàm gen_signal đã được chỉnh sửa để tạo quỹ đạo bán hình tròn
def gen_signal(t, phase, extension_amplitude=0.5, swing_amplitude=0.3, frequency=1.0):
    """Generates a modified sinusoidal reference leg trajectory with half-circle shape.

    Args:
      t: Current time in simulation.
      phase: The phase offset for the periodic trajectory.
      extension_amplitude: Amplitude for the leg extension (x-axis).
      swing_amplitude: Amplitude for the leg swing (y-axis).
      frequency: Frequency of the leg movement.

    Returns:
      The desired leg extension and swing angle at the current time.
    """
    period = 1 / frequency
    leg_theta = (2 * np.pi / period * t + phase) % (2 * np.pi)  # Calculate the angle in the cycle and wrap it to [0, 2π)

    # Calculate extension (x) as usual
    extension = -0.04 * np.cos(leg_theta)

    print(leg_theta)

    y_center = -0.27

    # Calculate swing (y) based on the half-circle logic
    if leg_theta > np.pi:  # In the second half of the cycle
        swing = y_center  # Keep y (swing) constant
    else:  # In the first half of the cycle
        swing = 0.08 * np.sin(leg_theta) + y_center

    return extension, swing


# Khởi tạo robot
robot = solo12_pybullet.Solo12PybulletEnv(on_rack=True)
kin = solo12_kinematic.Solo12Kinematic()
print(robot.build_motor_id_list())

steps = 20000  # Số bước để chạy thử
time_step = 0.01
frequency = 0.08  # Tần số bước chân
amplitude = 0.5
speed = 7

# Chạy mô phỏng
for step_counter in range(steps):
    t = step_counter * time_step

    # Tạo tín hiệu cho một chân (ví dụ: chân đầu tiên) với phase = 0
    extension, swing = gen_signal(t, phase=0, extension_amplitude=amplitude, swing_amplitude=amplitude / 2,
                                  frequency=frequency)

    motor_knee, motor_hip, _ = kin.inverse_kinematics(extension, swing, 0, 1)
    motor_hip += np.pi / 2
    # Sử dụng tín hiệu này để tạo động tác cho robot
    action = [0] * 12  # Gán giá trị vào hành động của chân đầu tiên, các chân khác giữ nguyên
    action[6] = motor_hip
    action[7] = motor_knee

    robot.step(action)

print("Simulation finished.")
