import numpy as np
from motion_imitation.robots import robot_config
from motion_imitation.robots import solo12

robot = solo12.Solo12(render=True, on_rack=False)

motor_control_mode = robot_config.MotorControlMode.POSITION

steps = 3000
amplitude = 0.3
speed = 5

init_motor_angles = np.array([-0.6, 0.7, -1.4,
                              0.6, 0.7, -1.4,
                              -0.6, -0.7, 1.4,
                              0.6, -0.7, 1.4])

foot = [[0.19795216216215378, -0.27987725202268765, -0.15702548693175464],
        [0.19793329856123945, 0.2798794695146239, -0.15702455914559812],
        [-0.196877889527097, -0.2824444622509925, -0.15697431956648367],
        [-0.19689313841606979, 0.2824450913303018, -0.15697353521914112]]

for i in range(4):
    print(robot.ComputeMotorAnglesFromFootLocalPosition(i, foot[i]))
