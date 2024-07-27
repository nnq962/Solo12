from Robot import solo12_pybullet

robot = solo12_pybullet.Solo12PybulletEnv(on_rack=False)
a, b = robot.build_motor_id_list()
print(a)
print(b)
while True:
    robot.step()

