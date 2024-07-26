import pybullet as p
import pybullet_data
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
num_joints = p.getNumJoints(robot_id)
joint_info = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in range(num_joints)}

# Đặt vận tốc cho khớp 'motor_knee_fr'
knee_joint_id = joint_info['motor_knee_fr']
p.setJointMotorControl2(bodyUniqueId=robot_id,
                        jointIndex=knee_joint_id,
                        controlMode=p.VELOCITY_CONTROL,
                        force=0)
p.setJointMotorControl2(bodyUniqueId=robot_id,
                        jointIndex=knee_joint_id,
                        controlMode=p.POSITION_CONTROL,
                        force=0)

# Áp dụng momen điều khiển
p.setJointMotorControl2(bodyUniqueId=robot_id,
                        jointIndex=knee_joint_id,
                        controlMode=p.TORQUE_CONTROL,
                        force=0.5)

# Chạy mô phỏng trong một khoảng thời gian để quan sát kết quả
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)

# Ngắt kết nối mô phỏng
p.disconnect()
