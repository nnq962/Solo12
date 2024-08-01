from Robot import solo12_pybullet

robot = solo12_pybullet.Solo12PybulletEnv(on_rack=True)
while True:
    # robot.p.resetDebugVisualizerCamera(0.95, 50, -20, robot.get_base_pos_and_orientation()[0])
    robot.step()

