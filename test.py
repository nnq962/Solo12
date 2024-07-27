import pybullet as p
import pybullet_data
import numpy as np
import time

# Kết nối với mô phỏng PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Đường dẫn tới các tệp dữ liệu URDF

# Tải mô hình URDF
robot_id = p.loadURDF("Robot/Simulation/solo12.urdf", [0, 0, 0.5])
p.createConstraint(robot_id,
                   -1, -1, -1,
                   p.JOINT_FIXED,
                   [0, 0, 0], [0, 0, 0], [0, 0, 0.4])
# Thiết lập trọng lực
p.setGravity(0, 0, -9.81)

# Đặt chế độ mô phỏng thực
p.setRealTimeSimulation(0)

# Liệt kê các khớp trong mô hình
# num_joints = p.getNumJoints(robot_id)
# joint_info = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in range(num_joints)}


num_joints = p.getNumJoints(robot_id)
joint_name_to_id = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
motor_names = ["motor_hip_fl",
               "motor_knee_fl",
               "motor_abduction_fl",

               "motor_hip_hr",
               "motor_knee_hr",
               "motor_abduction_hr",

               "motor_hip_fr",
               "motor_knee_fr",
               "motor_abduction_fr",

               "motor_hip_hl",
               "motor_knee_hl",
               "motor_abduction_hl"]
motor_id_list = [joint_name_to_id[motor_name] for motor_name in motor_names]

# Đặt vận tốc cho khớp 'motor_knee_fr'
# knee_joint_id = joint_info['motor_knee_fr']
# p.setJointMotorControl2(bodyUniqueId=robot_id,
#                         jointIndex=knee_joint_id,
#                         controlMode=p.VELOCITY_CONTROL,
#
#                         force=0)

# {'motor_abduction_fr': 0, 'motor_hip_fr': 1, 'motor_knee_fr': 2, 'motor_abduction_fl': 3, 'motor_hip_fl': 4,
# 'motor_knee_fl': 5, 'motor_abduction_hr': 6, 'motor_hip_hr': 7, 'motor_knee_hr': 8, 'motor_abduction_hl': 9,
# 'motor_hip_hl': 10, 'motor_knee_hl': 11}
# [4, 5, 3, 7, 8, 6, 1, 2, 0, 10, 11, 9]
theta1 = np.radians(132.969 - 90)
theta2 = np.radians(180 - 56.179 - (90 - np.degrees(theta1)))
target_angles = [0, 0, 0, 0, 0, 0, -theta1, theta2, 0, 0, 0, 0]
for motor_id, angle in zip(motor_id_list, target_angles):
    p.setJointMotorControl2(
        bodyIndex=robot_id,
        jointIndex=motor_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=angle,
        force=10)

# Áp dụng momen điều khiển
# p.setJointMotorControl2(bodyUniqueId=robot_id,
#                         jointIndex=knee_joint_id,
#                         controlMode=p.TORQUE_CONTROL,
#                         force=0.5)

# ------------------------------------------------------------------------------
# IN RA CÁC THÀNH PHẦN CỦA ROBOT
# In tên của thân chính (base link)
base_name = p.getBodyInfo(robot_id)[0].decode('utf-8')
print(f"Link index: -1, Link name: {base_name}")

# Lặp qua các link để in tên và chỉ số của chúng
num_joints = p.getNumJoints(robot_id)

for link_index in range(num_joints):
    link_name = p.getJointInfo(robot_id, link_index)[12].decode('utf-8')
    print(f"Link index: {link_index}, Link name: {link_name}")
# ------------------------------------------------------------------------------

# Chạy mô phỏng trong một khoảng thời gian để quan sát kết quả
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)

# Ngắt kết nối mô phỏng
p.disconnect()
