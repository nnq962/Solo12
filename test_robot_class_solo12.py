import numpy as np
from motion_imitation.robots import robot_config
from motion_imitation.robots import solo12
robot = solo12.Solo12(render=True, on_rack=False)

motor_control_mode = robot_config.MotorControlMode.POSITION

steps = 3000
amplitude = 0.3
speed = 5

init_motor_angles = np.array([0, 0.7, -1.4,
                              0, -0.7, 1.4,
                              0, 0.7, -1.4,
                              0, -0.7, 1.4])

while True:
    robot.Step(init_motor_angles, motor_control_mode)
    print(robot.GetMotorAngles().tolist())
