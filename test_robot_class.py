from Robot import solo12_pybullet
import numpy as np
action_temp = [0] * 20
robot = solo12_pybullet.Solo12PybulletEnv(on_rack=False)
steps = 20000000
action = [0] * 12
amplitude = 0.5
speed = 7
print(robot.build_motor_id_list())
for step_counter in range(steps):
    time_step = 0.01
    t = step_counter * time_step
    action[1] = action[7] = (np.sin(speed * t) * amplitude + np.pi / 5)
    action[4] = action[10] = -action[1]
    robot.step(action)

