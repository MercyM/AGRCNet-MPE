import numpy as np
import torch
import os
from policy.alg_maddpg_baseline import MADDPG
from common.misc import gumbel_softmax, onehot_from_logits

class AgentBaseline:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)
        self.exploration = 0.3

    def scale_noise(self, scale):
        self.exploration = scale

    def select_action(self, o, noise_rate, epsilon):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        action = self.policy.actor_network(inputs)
        actions = gumbel_softmax(action, hard=True).squeeze(0)
        actions = actions.data.numpy()

        return actions.copy()

    def learn(self, transitions, other_agents,r):
        self.policy.train(transitions, other_agents,r)
