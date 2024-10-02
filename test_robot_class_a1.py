from pybullet_utils import bullet_client
import pybullet_data
import pybullet
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
import numpy as np

p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
p.setPhysicsEngineParameter(numSolverIterations=30)
p.setTimeStep(0.001)
p.setGravity(0, 0, -9.8)
p.setPhysicsEngineParameter(enableConeFriction=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

robot = a1.A1(p, on_rack=True)
motor_control_mode = robot_config.MotorControlMode.POSITION

init_motor_angles = np.array([-0.3, 1, -1.8,
                              0.3, 1, -1.8,
                              -0.3, 1, -1.8,
                              0.3, 1, -1.8])

steps = 10000
amplitude = 0.5
speed = 5

actions_and_observations = []

for step_counter in range(steps):
    # Matches the internal timestep.
    time_step = 0.01
    t = step_counter * time_step

    action = [np.sin(speed * t) * amplitude + 0] * 12
    action[2] = action[5] = action[8] = action[11] = action[1] = action[4] = action[7] = action[10] = 0
    action = np.array(action)
    print(action.reshape(4, 3))

    robot.Step(action, motor_control_mode)