import random
import numpy as np
from collections import deque
import gym
import torch


class Memory:
    """ Relay buffer class """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self):
        return len(self.buffer)


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Action space must be of type Box"
        self.action_space = env.action_space

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def save_models_for_testing(a, filename):
    torch.save({
        'actor_state_dict': a.actor.state_dict(),
        'critic_state_dict': a.critic.state_dict(),
    }, filename)


def save_models_for_training(a, filename):
    torch.save({
        'actor_state_dict': a.actor.state_dict(),
        'critic_state_dict': a.critic.state_dict(),
        'actor_optimizer_state_dict': a.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': a.critic_optimizer.state_dict(),
    }, filename)


def load_models_for_testing(a, filename):
    checkpoint = torch.load(filename)
    a.actor.load_state_dict(checkpoint['actor_state_dict'])
    a.critic.load_state_dict(checkpoint['critic_state_dict'])
    update_target_networks(a)


def load_models_for_training(a, filename):
    checkpoint = torch.load(filename)
    a.actor.load_state_dict(checkpoint['actor_state_dict'])
    a.critic.load_state_dict(checkpoint['critic_state_dict'])
    a.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    a.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    update_target_networks(a)


def update_target_networks(a):
    for target_param, param in zip(a.actor_target.parameters(), a.actor.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(a.critic_target.parameters(), a.critic.parameters()):
        target_param.data.copy_(param.data)
