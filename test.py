import pybullet as p
import pybullet_data
import numpy as np
import time

# Tạo kết nối với PyBullet
client = p.connect(p.GUI)

# Tải dữ liệu URDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Đặt trọng lực
p.setGravity(0, 0, -9.8)

# Tạo mặt phẳng
p.loadURDF("plane.urdf")

# Tạo robot (ở đây sử dụng Kuka Arm, bạn có thể thay thế bằng robot khác nếu có URDF)
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1], useFixedBase=False)

# Chạy mô phỏng
p.setRealTimeSimulation(0)  # Tắt mô phỏng thời gian thực


# Hàm để biến đổi vận tốc góc từ không gian toàn cục sang không gian cục bộ
def transform_angular_velocity_to_local_frame(angular_velocity, orientation):
    _, orientation_inversed = p.invertTransform([0, 0, 0], orientation)
    relative_velocity, _ = p.multiplyTransforms([0, 0, 0], orientation_inversed, angular_velocity, [0, 0, 0, 1])
    return np.array(relative_velocity)


# Vòng lặp mô phỏng
for step in range(1000):
    p.stepSimulation()  # Chạy một bước mô phỏng

    # Lấy vận tốc góc và vị trí robot
    linear_velocity, angular_velocity_global = p.getBaseVelocity(robot_id)

    # Lấy tư thế (quaternion) hiện tại của robot
    position, orientation = p.getBasePositionAndOrientation(robot_id)

    # Biến đổi vận tốc góc từ không gian toàn cục sang không gian cục bộ
    angular_velocity_local = transform_angular_velocity_to_local_frame(angular_velocity_global, orientation)

    # In ra vận tốc góc toàn cục và cục bộ
    print(f"Step {step}:")
    print(f"  Global Angular Velocity: {angular_velocity_global}")
    print(f"  Local Angular Velocity:  {angular_velocity_local}")

    time.sleep(0.01)  # Để quan sát dễ hơn

# Ngắt kết nối PyBullet
p.disconnect()
