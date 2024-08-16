import numpy as np

from Robot import solo12_pybullet


robot = solo12_pybullet.Solo12PybulletEnv(on_rack=False)
action = [0.0] * 12
action = np.array(action)
steps = 20000
for i in range(steps):
    robot.step(action)
