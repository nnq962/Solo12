from Robot import solo12_pybullet
import matplotlib.pyplot as plt
from DDPG.DDPG_model import *
import sys
from DDPG.utils import *

steps = 4000
episodes = 3000

env = solo12_pybullet.Solo12PybulletEnv(
    render=False,
    end_steps=steps,
    on_rack=False,
    imu_noise=True,
    pd_control_enabled=True
)

# env = NormalizedEnv(gym.make("Pendulum-v0"))

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

    if (episode + 1) % 50 == 0:
        torch.save({
            'episode': episode + 1,
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        }, f'checkpoint_episode_{episode + 1}.pth')
        print(f"Model saved at episode {episode + 1}")

torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, "ddpg_agent_final.pth")

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
