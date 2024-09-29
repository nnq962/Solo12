from motion_imitation.robots import a1
from motion_imitation.robots import solo12
import pybullet_data
from pybullet_utils import bullet_client
import pybullet  # pytype:disable=import-error

test = True
if test:
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
else:
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
p.setPhysicsEngineParameter(numSolverIterations=30)
p.setTimeStep(0.001)
p.setGravity(0, 0, -9.8)
p.setPhysicsEngineParameter(enableConeFriction=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

a1_robot = solo12.Solo12(p, on_rack=False)

while True:
    a1_robot._SettleDownForReset(None, 10)
    p.stepSimulation()