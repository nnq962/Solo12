from Robot import solo12_pybullet
import numpy as np
action_temp = [0] * 20
robot = solo12_pybullet.Solo12PybulletEnv(on_rack=False)

while True:
    robot.step(action_temp)

