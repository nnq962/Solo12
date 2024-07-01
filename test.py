import pybullet as p
import pybullet_data
import sys

# Khởi tạo môi trường PyBullet
physicsClient = p.connect(p.GUI)  # Mở môi trường với GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)

# Kiểm tra xem có đường dẫn được cung cấp qua dòng lệnh không
if len(sys.argv) < 2:
    print("Cách sử dụng: python3 test_urdf.py <đường dẫn tới file URDF>")
    exit(1)

# Lấy đường dẫn file URDF từ dòng lệnh
urdf_path = sys.argv[1]

# Load robot từ file URDF
robot_urdf_path = urdf_path  # Đường dẫn tới file URDF của bạn
robot_start_pose = [0, 0, 1]  # Vị trí ban đầu của robot
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Hướng ban đầu của robot (ở đây là không xoay)
robot_id = p.loadURDF(robot_urdf_path, robot_start_pose, robot_start_orientation)

# Load sàn hoặc giá để đặt robot lên đó
plane_id = p.loadURDF("plane.urdf")  # Load sàn hoặc giá
# p.createConstraint(
#                 robot_id, -1, -1, -1, p.JOINT_FIXED,
#                 [0, 0, 0], [0, 0, 0], [0, 0, 0.4])

while True:
    p.stepSimulation()
