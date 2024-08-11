from Robot import solo12_pybullet
action_temp = [0] * 20
robot = solo12_pybullet.Solo12PybulletEnv(on_rack=True)
while True:

    keys = robot.p.getKeyboardEvents()
    if robot.p.B3G_LEFT_ARROW in keys and keys[robot.p.B3G_LEFT_ARROW] & robot.p.KEY_WAS_TRIGGERED:
        print("Left arrow pressed")
        print("------------------")
        # Thêm logic để điều khiển robot sang trái
    elif robot.p.B3G_RIGHT_ARROW in keys and keys[robot.p.B3G_RIGHT_ARROW] & robot.p.KEY_WAS_TRIGGERED:
        print("Right arrow pressed")
        print("------------------")
        # Thêm logic để điều khiển robot sang phải
    elif robot.p.B3G_UP_ARROW in keys and keys[robot.p.B3G_UP_ARROW] & robot.p.KEY_WAS_TRIGGERED:
        print("Up arrow pressed")
        print("------------------")
        robot.step_length = 0.09
        # Thêm logic để điều khiển robot tiến tới
    elif robot.p.B3G_DOWN_ARROW in keys and keys[robot.p.B3G_DOWN_ARROW] & robot.p.KEY_WAS_TRIGGERED:
        print("Down arrow pressed")
        print("------------------")
        robot.step_length = -0.09
        # Thêm logic để điều khiển robot lùi lại
    elif ord('q') in keys and keys[ord('q')] & robot.p.KEY_WAS_TRIGGERED:
        print("Q pressed")
        print("------------------")
        break
    robot.step(action=action_temp)

