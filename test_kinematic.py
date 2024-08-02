from Robot import solo12_kinematic
import numpy as np

kine1 = solo12_kinematic.Solo12Kinematic()

x = 0.1
y = -0.25
z = 0.05

print(np.degrees(kine1.inverse_kinematics(x, y, z, 1)))

"""
[0, -0.25, 0.05]    -> [74.36362459 -127.1818123    11.30993247]
[0, -0.25, 0]       -> [77.24966575 -128.62483287    0.]
"""
