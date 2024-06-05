import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, args, hidden_dim=128, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Critic, self).__init__()
        self.args = args
        self.s_dim = args.obs_shape[0]
        self.a_dim = args.action_shape[0]
        self.n_agents = args.num_adversaries
        input_dim = (args.obs_shape[0]) + (args.action_shape[0])
        out_dim = 1
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, state, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        # action = torch.stack(action, dim=0)
        # state = torch.stack(state, dim=0)
        x = torch.cat((state, action), dim=-1)
        x = x.reshape(-1, self.s_dim + self.a_dim)
        h = self.in_fn(x)
        h1 = self.nonlin(self.fc1(h))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        out = out.reshape(self.n_agents, -1, 1)
        return out

class ActorPart1(nn.Module):
    def __init__(self, args):
        """
        Arguments:
            hidden_size: the size of the output
            num_inputs: the size of the input -- (batch_size*nagents, obs_shape)
        Output:
            x: individual thought -- (batch_size*nagents, hidden_size)
        """
        super(ActorPart1, self).__init__()
        self.args = args
        self.s_dim = args.obs_shape[0]
        self.a_dim = args.action_shape[0]
        self.n_agents = args.num_adversaries
        hidden_size = args.hidden_size

        self.linear1 = nn.Linear(self.s_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, observation):
        x = observation
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        return x
        # returns "individual thought", since this will go into the Attentional Unit


class AttentionUnit(nn.Module):
    # Currently using MLP, later try LSTM
    def __init__(self, args):
        # a binary classifier
        # num_inputs is the size of "thoughts"
        super(AttentionUnit, self).__init__()
        num_output = 1
        self.args = args
        hidden_size = args.hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_output)

    def forward(self, thoughts):
        # thoughts is the output of actor_part1
        x = self.linear1(thoughts)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        output = torch.sigmoid(x)
        return output


class CommunicationChannel(nn.Module):
    def __init__(self, args):
        """
        Arguments:
            hidden_size: the size of the "thoughts"
        """
        self.args = args
        hidden_size = args.hidden_size // 2
        super(CommunicationChannel, self).__init__()
        self.bi_GRU = nn.GRU(args.hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, init_hidden):
        """
        Arguments:
            inputs: "thoughts"  -- (batch_size, seq_len, actor_hidden_size)
        Output:
            x: intergrated thoughts -- (batch_size, seq_len, num_directions * hidden_size)
        """
        x = self.bi_GRU(inputs, init_hidden)
        return x


class ActorPart2(nn.Module):
    def __init__(self, args):
        """
        Arguments:
            hidden_size: the size of the output
            num_inputs: the size of the obs -- (batch_size*nagents, obs_shape)
        Output:
            x: individual action -- (batch_size*nagents, action_shape)
        """
        super(ActorPart2, self).__init__()

        self.args = args
        self.s_dim = args.obs_shape[0]
        self.a_dim = args.action_shape[0]
        self.n_agents = args.num_adversaries
        hidden_size = args.hidden_size
        # num_outputs = action_space.n

        self.linear1 = nn.Linear(self.s_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.a_dim)

    def forward(self, thoughts, padding):
        x = thoughts + padding
        x = F.relu(x)
        # x = F.relu(self.ln1(self.linear1(x)))
        x = torch.sigmoid(self.linear2(x))
        output = x.reshape(-1, self.args.action_shape[0])
        return output