import torch.autograd
import torch.optim as optim
from DDPG.network import *
from DDPG.utils import *
from torch.autograd import Variable


class DDPGagent:
    def __init__(self, env, hidden_size1=256, hidden_size2=128, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
                 max_memory_size=100000):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = ActorNetwork(self.num_states, hidden_size1, hidden_size2, self.num_actions)
        self.actor_target = ActorNetwork(self.num_states, hidden_size1, hidden_size2, self.num_actions)
        self.critic = CriticNetwork(self.num_states + self.num_actions, hidden_size1, hidden_size2, self.num_actions)
        self.critic_target = CriticNetwork(self.num_states + self.num_actions, hidden_size1, hidden_size2, self.num_actions)

        # Synchronize Target Networks with Original Networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        q_vals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_q = self.critic_target.forward(next_states, next_actions.detach())
        q_prime = rewards + self.gamma * next_q
        critic_loss = self.critic_criterion(q_vals, q_prime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


