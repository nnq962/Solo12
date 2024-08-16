import numpy as np

from Robot import solo12_pybullet


action = [0.0] * 12
action = np.array(action)
steps = 2000
total_reward = 0
robot = solo12_pybullet.Solo12PybulletEnv(on_rack=False, end_steps=steps)

for i in range(steps):
    state, reward, done, _ = robot.step(action)
    total_reward += reward

print("Total reward: ", total_reward)
