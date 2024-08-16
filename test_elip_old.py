from Robot import solo12_pybullet


robot = solo12_pybullet.Solo12PybulletEnv(on_rack=False)
action = [0] * 12

while True:
    robot.step(action)
    robot.pybullet_client.resetDebugVisualizerCamera(1.0, 45, -40, robot.get_base_pos_and_orientation()[0])
