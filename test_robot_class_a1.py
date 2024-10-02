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

robot = a1.A1(p, on_rack=False)
motor_control_mode = robot_config.MotorControlMode.POSITION

init_motor_angles = np.array([-0.3, 1, -1.8,
                              0.3, 1, -1.8,
                              -0.3, 1, -1.8,
                              0.3, 1, -1.8])

while True:
    robot.Step(init_motor_angles, motor_control_mode)
    robot.ComputeMotorAnglesFromFootLocalPosition(0, [0.14544602, - 0.2035496, - 0.21173298])
