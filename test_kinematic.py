from Robot import solo12_kinematic_2
from Robot import solo12_kinematic
import numpy as np

kine1 = solo12_kinematic.Solo12Kinematic()
kine2 = solo12_kinematic_2.Solo12Kinematic()

x = 0.05
y = -0.25
z = 0

print(np.degrees(kine1.inverse_kinematics(x, y, z, 0)))
print(np.degrees(kine2.inverse_kinematics(x, y, z)))
