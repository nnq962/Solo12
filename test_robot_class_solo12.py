import numpy as np
from motion_imitation.robots import robot_config
from motion_imitation.robots import solo12

robot = solo12.Solo12(render=True, on_rack=False)
motor_control_mode = robot_config.MotorControlMode.POSITION


def gen_signal(t, phase):
    period = 1 / 2.5
    theta = (2 * np.pi / period * t + phase) % (2 * np.pi)

    x = -0.08 * np.cos(theta) + 0
    if theta > np.pi:
        y = -0.23
    else:
        y = 0.06 * np.sin(theta) - 0.23
    z = y
    y = robot.kinematic.l_hip
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

    motors_fr = robot.kinematic.inverse_kinematics(foot_1)

    return motors_fr


steps = 10000
amplitude = 0.5
speed = 5

actions_and_observations = []
init_motor_angles = np.array([0, 0.8, -1.6,
                              0, 0.8, -1.6,
                              0, -0.8, 1.6,
                              0, -0.8, 1.6])


for step_counter in range(steps):

    robot.Step(init_motor_angles, motor_control_mode)
