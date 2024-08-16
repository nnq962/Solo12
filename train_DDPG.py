from Robot import solo12_pybullet
import matplotlib.pyplot as plt
from DDPG.DDPG_model import *
import sys
from DDPG.utils import *

steps = 4000
episodes = 2000

env = solo12_pybullet.Solo12PybulletEnv(
    render=False,
    end_steps=steps,
    on_rack=False,
    imu_noise=True,
    pd_control_enabled=True
)

agent = DDPGagent(env)
noise = OUNoise(env.action_space)
batch_size = 256
rewards = []
avg_rewards = []

for episode in range(episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(steps):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2),
                                                                        np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

    # Save models every 200 episodes
    # if episode % 200 == 0:
    #     save_models_for_training(agent, f"ddpg_agent_{episode}.pth")

save_models_for_testing(agent, "ddpg_agent_final.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
