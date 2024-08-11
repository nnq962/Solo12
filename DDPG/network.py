import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.autograd


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Parameters for Critic network
        :param input_size: state size + action size
        :param hidden_size: hidden size
        :param output_size: output size
        """
        super(CriticNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

