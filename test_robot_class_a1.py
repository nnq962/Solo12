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

def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos(
        (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
        (2 * l_low * l_up))
    l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])

def gen_signal(t, phase):
    period = 1 / 0.6
    theta = (2 * np.pi / period * t + phase) % (2 * np.pi)

    x = -0.08 * np.cos(theta) - 0
    if theta > np.pi:
        y = -0.23
    else:
        y = 0.07 * np.sin(theta) - 0.26
    z = y
    y = 0.08505
    return [x, y, z]

def signal(t):
    """Generates the trotting gait for the robot.

    Args:
      t: Current time in simulation.

    Returns:
      A numpy array of the reference leg positions.
    """
    # Generates the leg trajectories for the two digonal pair of legs.
    foot_1 = gen_signal(t, phase=0)
    foot_2 = gen_signal(t, phase=np.pi)

    motors_fr = foot_position_in_hip_frame_to_joint_angle(foot_1)
    motors_fl = foot_position_in_hip_frame_to_joint_angle(foot_2)
    motors_hr = foot_position_in_hip_frame_to_joint_angle(foot_2)
    motors_hl = foot_position_in_hip_frame_to_joint_angle(foot_1)
    a = np.concatenate((motors_fr, motors_fl, motors_hr, motors_hl))

    return a

action = np.zeros(12)

while True:
    action = signal(robot.GetTimeSinceReset())
    robot.Step(action, motor_control_mode)
