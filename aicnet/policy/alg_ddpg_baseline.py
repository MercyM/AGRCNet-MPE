import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import os
from net.actor_critic_ddpg_baseline import Actor, Critic
import tensorflow as tf
import numpy as np

sess = tf.Session()


class DDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args, agent_id)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args, agent_id)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + 'baseline' + '/' + 'ddpg_baseline' + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

        self.summary_ops, self.summary_vars = build_summaries()
        sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(
            self.args.save_dir + '/' + 'baseline' + '/' + 'ddpg_baseline' + '/' + '/loss' + '/' + 'agent_%d' % agent_id,
            graph=tf.get_default_graph())

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * param.data + self.args.tau * target_param.data)

    # update the network
    def train(self, transitions, other_agents, reward):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o = transitions['o_%d' % self.agent_id]
        u = transitions['u_%d' % self.agent_id]
        o_next = transitions['o_next_%d' % self.agent_id]

        # calculate the target Q value function
        with torch.no_grad():
            # 得到下一个状态对应的动作

            u_next = self.actor_target_network(o_next)

            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u = self.actor_network(o)
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

        summary_str = sess.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: reward[self.agent_id],
            # self. summary_vars[1]: q_show,
            self.summary_vars[2]: critic_loss.detach().numpy(),
            self.summary_vars[3]: actor_loss.detach().numpy()
        })
        self.writer.add_summary(summary_str, self.train_step)
        self.writer.flush()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = self.args.save_dir + '/' + 'baseline' + '/' + 'ddpg_baseline' + '/' + self.args.scenario_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + num + '_critic_params.pkl')


def build_summaries():
    episode_reward = tf.Variable(0., name="episode_reward")
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0., name="episode_ave_max_q")
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    c_loss = tf.Variable(0., name="critic_loss")
    tf.summary.scalar("Critic_loss", c_loss)
    a_loss = tf.Variable(0., name="actor_loss")
    tf.summary.scalar("Actor_loss", a_loss)

    summary_vars = [episode_reward, episode_ave_max_q, c_loss, a_loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
