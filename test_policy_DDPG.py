from Robot import solo12_pybullet
from DDPG.DDPG_model import *
import torch

# Thử nghiệm với 10 episodes
num_episodes = 30
steps = 4000
results = []


# Khởi tạo môi trường
env = solo12_pybullet.Solo12PybulletEnv(render=True,
                                        pd_control_enabled=True,
                                        on_rack=True,
                                        end_steps=steps)


# Khởi tạo agent và tải mô hình đã được huấn luyện
agent = DDPGagent(env)
load_models_for_testing(agent, "ddpg_agent_final.pth")

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    with torch.no_grad():  # Tắt gradient để tiết kiệm tài nguyên
        for step in range(steps):
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if step % 100 == 0:
                print(f"Episode {episode+1}, Step: {step}, Current Reward: {reward}")

            if done:
                print(f"Episode {episode+1} finished after {step+1} steps with total reward: {total_reward}")
                results.append(total_reward)
                break

# Lưu kết quả vào file nếu cần
# with open("test_results.txt", "w") as f:
#     for result in results:
#         f.write(f"{result}\n")

print("Testing completed.")
