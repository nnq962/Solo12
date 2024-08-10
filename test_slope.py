import pybullet as p
import pybullet_data
import time

# Khởi tạo PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Tải sàn nhà
p.loadURDF("plane.urdf")

# Tải độ dốc từ tệp URDF
slope_body = p.loadURDF("parametric_slope.urdf", basePosition=[0, 0, 0])

# Thiết lập trọng lực
p.setGravity(0, 0, -9.81)

# Mô phỏng trong 5 giây
for i in range(500000):
    p.stepSimulation()
    time.sleep(1./240.)

# Ngắt kết nối PyBullet
p.disconnect()
