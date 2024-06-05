import numpy as np
import torch
import os
from policy.alg_try import TRY
from common.misc import gumbel_softmax, onehot_from_logits


class Agent_try:
    def __init__(self, args):
        self.args = args
        self.policy = TRY(args)

    def select_action(self, o, noise_rate, epsilon):
        actions = []
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)  # [1, 3, 16]
        pi, self.hidden = self.policy.actor_network(inputs, self.hidden, False)
        for agent_id in range(self.args.num_adversaries):
            u = gumbel_softmax(pi[agent_id], hard=True).squeeze(0)
            actions.append(u.data.numpy())
        return actions.copy()

    def learn(self, transitions):
        self.policy.train(transitions)

    def select_action2(self, thoughts, padding, C):
        nagents = thoughts.shape[0]

        # merge invidual thoughts and intergrated thoughts
        is_comm = C.any(dim=0)  # (nagents)
        # agent withouth communication padding with zeros
        for i in range(nagents):
            if not is_comm[i]:
                padding[i] = 0

        actions = []
        # input to part II of the actor
        pi = self.policy.actor_p2(thoughts, padding)
        for agent_id in range(self.args.num_adversaries):
            u = gumbel_softmax(pi[agent_id].unsqueeze(0), hard=True).squeeze(0)
            actions.append(u.data.numpy())
        return actions.copy()

    def initiate_group(self, obs_n, m, thoughts):
        return self.policy.initiate_group(obs_n, m, thoughts)

    def get_thoughts(self, obs_n):
        return self.policy.get_thoughts(obs_n)

    def update_thoughts(self, thoughts, C):
        return self.policy.update_thoughts(thoughts, C)
