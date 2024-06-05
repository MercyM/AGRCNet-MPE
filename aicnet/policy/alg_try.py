import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import os
from net.actor_critic_try import Critic, ActorPart1, ActorPart2, AttentionUnit, CommunicationChannel
import tensorflow as tf
import numpy as np
from common.misc import gumbel_softmax, onehot_from_logits
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import queue

MSELoss = torch.nn.MSELoss()
sess = tf.Session()


class TRY:
    def __init__(self, args):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.train_step = 0
        # create the network
        self.critic_network = Critic(args)
        self.actor_p1 = ActorPart1(args)
        self.atten = AttentionUnit(args)
        self.comm = CommunicationChannel(args)
        self.actor_p2 = ActorPart2(args)

        # build up the target network
        self.critic_target_network = Critic(args)
        self.actor_target_p1 = ActorPart1(args)
        self.comm_target = CommunicationChannel(args)
        self.actor_target_p2 = ActorPart2(args)

        # load the weights into the target networks
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_target_p1.load_state_dict(self.actor_p1.state_dict())
        self.comm_target.load_state_dict(self.comm.state_dict())
        self.actor_target_p2.load_state_dict(self.actor_p2.state_dict())

        # create the optimizer
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.atten_optim = torch.optim.Adam(self.atten.parameters(), lr=self.args.lr_actor)
        self.comm_optim = torch.optim.Adam(self.comm.parameters(), lr=self.args.lr_actor)
        self.actor_optim = torch.optim.Adam([
            {'params': self.actor_p1.parameters(), 'lr': self.args.lr_actor},
            {'params': self.actor_p2.parameters(), 'lr': self.args.lr_actor}
        ])

        self.comm_hidden_size = self.args.hidden_size // 2
        # replay for the update of attention unit
        self.queue = queue.Queue()

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + 'result' + '/' + 'try' + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/' + '420_actor_params.pkl'):
            # self.actor_network.load_state_dict(torch.load(self.model_path + '/' + '420_actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/' + '420_critic_params.pkl'))
            print('successfully loaded actor_network: {}'.format(self.model_path + '/' + '420_actor_params.pkl'))
            print('successfully loaded critic_network: {}'.format(self.model_path + '/' + '420_critic_params.pkl'))

        self.summary_ops, self.summary_vars = build_summaries()
        sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(
            self.args.save_dir + '/' + 'result' + '/' + 'try' + '/' + '/loss' + '/' + 'agent',
            graph=tf.get_default_graph())

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

        for target_param, param in zip(self.actor_target_p1.parameters(), self.actor_p1.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

        for target_param, param in zip(self.comm_target.parameters(), self.comm.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

        for target_param, param in zip(self.actor_target_p2.parameters(), self.actor_p2.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)


    # update the network
    def train(self, transitions):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)

        o = transitions['o']
        u = transitions['u']
        o_next = transitions['o_next']
        r = transitions['r']
        done = transitions['done']
        C = transitions['C']

        o_input = torch.tensor(np.array([item.numpy() for item in o])).reshape(self.args.batch_size,
                                                                               self.args.num_adversaries, -1)
        u_input = torch.tensor(np.array([item.numpy() for item in u])).reshape(self.args.batch_size,
                                                                               self.args.num_adversaries, -1)
        o_next_input = torch.tensor(np.array([item.numpy() for item in o_next])).reshape(self.args.batch_size,
                                                                                         self.args.num_adversaries, -1)
        r_input = torch.tensor(np.array([item.numpy() for item in r])).reshape(self.args.batch_size,
                                                                               self.args.num_adversaries, -1)
        done_input = torch.tensor(np.array([item.numpy() for item in done])).reshape(self.args.batch_size,
                                                                                     self.args.num_adversaries, -1)
        # C_batch = np.array([item.numpy() for item in C]).reshape(self.args.batch_size, self.args.num_adversaries, -1)
        C_batch = torch.BoolTensor(np.array([item.numpy() for item in C])).reshape(self.args.batch_size,
                                                                                   self.args.num_adversaries, -1)
        # calculate the target Q value function
        target_Q = []
        Q = []
        for batch_index in range(self.args.batch_size):
            is_comm = C_batch[batch_index].any(dim=0)

            with torch.no_grad():
                next_thoughts_n = self.actor_target_p1(o_next_input[batch_index])
                # communication
                padding = torch.zeros_like(next_thoughts_n)
                for agent_i in range(self.args.num_adversaries):
                    if not C_batch[batch_index, agent_i, agent_i]: continue

                    # (1, m, actor_hiddensize)
                    thoughts_m = next_thoughts_n[C_batch[batch_index, agent_i]].unsqueeze(0)
                    hidden_state = torch.zeros((2 * 1, 1, self.comm_hidden_size))
                    inter_thoughts, _ = self.comm_target(thoughts_m, hidden_state)  # (1, m, 2*comm_hidden_size)
                    inter_thoughts = inter_thoughts.squeeze()  # (m, 2*comm_hiddensize)

                    # inter_thoughts = inter_thoughts.reshape(256, 4, 256, 64)
                    # TODO: Can this avoid in-place operation and pass the gradient?

                    padding[C_batch[batch_index, agent_i]] = inter_thoughts

                # select action for m agents with communicaiton

                next_thoughts_m = next_thoughts_n[is_comm]  # (m, actor_hiddensize)
                padding = padding[is_comm]  # (m, actor_hiddensize)
                reward_m = r_input[batch_index, is_comm]

                # print(padding.shape)
                next_action_m = self.actor_target_p2(next_thoughts_m, padding)  # (m, action_shape)
                next_obs_m = o_next_input[batch_index, is_comm]
                next_action = []
                for agent_i in range(self.args.num_adversaries):
                    next_action.append(onehot_from_logits(next_action_m[agent_i].unsqueeze(0)).squeeze(0))
                next_action = torch.stack(next_action, dim=0)
                next_Q_m = self.critic_target_network(next_obs_m, next_action)  # (m, 1)
                target_Q_m = reward_m + (self.args.gamma * next_Q_m).detach()

                # the q loss
            obs_m = o_input[batch_index, is_comm]
            action_m = u_input[batch_index, is_comm]

            q_value = self.critic_network(obs_m, action_m)

            target_Q.append(target_Q_m)
            Q.append(q_value)

        # the q loss
        target_Q = torch.stack(target_Q, dim=0)
        Q = torch.stack(Q, dim=0)
        critic_loss = MSELoss(Q, target_Q.detach())
        self.critic_optim.zero_grad()
        torch.nn.utils.clip_grad_norm(self.critic_network.parameters(), 0.5)
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        # the actor loss

        actor_loss = []
        for batch_index in range(self.args.batch_size):
            is_comm = C_batch[batch_index].any(dim=0)
            thoughts_n = self.actor_p1(o_input[batch_index])
            # communication
            padding = torch.zeros_like(thoughts_n)

            for agent_i in range(self.args.num_adversaries):
                if not C_batch[batch_index, agent_i, agent_i]: continue

                thoughts_m = thoughts_n[C_batch[batch_index, agent_i]].unsqueeze(0)  # (1, m, actor_hiddensize)
                hidden_state = torch.zeros((2 * 1, 1, self.comm_hidden_size))
                inter_thoughts, _ = self.comm(thoughts_m, hidden_state)
                inter_thoughts = inter_thoughts.squeeze()  # (m, 2*comm_hiddensize)

                # TODO: Can this avoid in-place operation and pass the gradient?
                padding[C_batch[batch_index, agent_i]] = inter_thoughts

            # select action for m agents with communication
            thoughts_m = thoughts_n[is_comm]
            padding = padding[is_comm]
            obs_m = o_input[batch_index, is_comm]  # (m, obs shape)
            action_m = self.actor_p2(thoughts_m, padding)  # (nagents, action shape)
            action = []
            for agent_i in range(self.args.num_adversaries):
                action.append(gumbel_softmax(action_m[agent_i].unsqueeze(0), hard=True).squeeze(0))
            action = torch.stack(action, dim=0)
            actor_loss_batch = -self.critic_network(obs_m, action)  # (m, 1)
            actor_loss.append(actor_loss_batch)

        actor_loss = torch.stack(actor_loss, dim=0)  # (batch_size, m, 1)
        actor_loss = actor_loss.mean()
        self.actor_optim.zero_grad()
        self.comm_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm(self.actor_network.parameters(), 0.5)
        self.actor_optim.step()
        self.comm_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

        summary_str = sess.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: critic_loss.detach().numpy(),
            self.summary_vars[1]: actor_loss.detach().numpy()
        })
        self.writer.add_summary(summary_str, self.train_step + 0)  # 这里要往上加
        self.writer.flush()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate + 0)  # 增加已有的
        model_path = self.args.save_dir + '/' + 'result' + '/' + 'try' + '/' + self.args.scenario_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + num + '_critic_params.pkl')


    def get_thoughts(self, obs_n):
        obs_n_tensor = torch.FloatTensor(obs_n)  # (nagents, obs_shape)
        thoughts = self.actor_p1(obs_n_tensor)
        return thoughts

    def initiate_group(self, obs_n, m, thoughts):
        obs_n = np.array(obs_n)
        nagents = obs_n.shape[0]

        # decide whether to initiate communication
        atten_out = self.atten(thoughts)  # (nagents, 1)
        is_comm = (atten_out > 0.5).squeeze()  # (nagents, )
        C = torch.zeros(nagents, nagents).bool()

        # relative position
        other_pos = (obs_n[:, -(nagents - 1) * 2:]).reshape(-1, nagents - 1, 2)  # (nagents, nagents-1, 2)
        other_dist = np.sqrt(np.sum(np.square(other_pos), axis=-1))  # (nagents, nagents-1)
        # insert itself distance into other_dist -> total_dist
        total_dist = []
        for i in range(nagents):
            total_dist.append(np.insert(other_dist[i], obj=i, values=0.0))
        total_dist = np.stack(total_dist)  # (nagents, nagents)
        # the id of top-m agents (including itself)
        index = np.argsort(total_dist, axis=-1)
        assert m <= nagents
        neighbour_m = index[:, :m]  # (nagents, m)

        for index, comm in enumerate(is_comm):
            if comm: C[index, neighbour_m[index]] = True

        # TODO: test the other parts of this project without attention unit
        C = torch.zeros(nagents, nagents)
        C[0] = 1
        C = C.bool()

        return C

    def update_thoughts(self, thoughts, C):
        nagents = thoughts.shape[0]
        thoughts = thoughts.clone().detach()

        for index in range(nagents):
            if not C[index, index]: continue
            input_comm = []
            # the neighbour of agent_i
            for j in range(nagents):
                if C[index, j]:
                    input_comm.append(thoughts[j])
            input_comm = torch.stack(input_comm, dim=0).unsqueeze(0)  # (1, m, acotr_hidden_size)
            # input communication channel to intergrate thoughts
            hidden_state = torch.zeros((2 * 1, 1, self.comm_hidden_size))
            intergrated_thoughts, _ = self.comm(input_comm, hidden_state)  # (1, m, 2*comm_hidden_size)
            intergrated_thoughts = intergrated_thoughts.squeeze()

            # update group_index intergrated thoughts
            thoughts[C[index]] = intergrated_thoughts

        return thoughts

def build_summaries():
    c_loss = tf.Variable(0., name="critic_loss")
    tf.summary.scalar("Critic_loss", c_loss)
    a_loss = tf.Variable(0., name="actor_loss")
    tf.summary.scalar("Actor_loss", a_loss)

    summary_vars = [c_loss, a_loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
