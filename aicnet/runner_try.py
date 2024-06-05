from tqdm import tqdm
from agents.agent_try import Agent_try as Agent
from agents.agent_baseline import AgentBaseline
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.adver_agents = self._init_adver_agents()
        self.good_agents = self._init_good_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + 'result' + '/' + 'try' + '/' + 'reward'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.returns = []

        # if os.path.exists(self.save_path + '/returns.pkl.npy'):
        #     returns_load = np.load(self.save_path + '/returns.pkl.npy')
        #     returns_load = returns_load.reshape(420)  # 这里要改
        #     self.returns = returns_load.tolist()

        self.summary_ops, self.summary_vars = build_summaries()
        sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(
            self.args.save_dir + '/' + 'result' + '/' + 'try' + '/' + 'reward',
            graph=tf.get_default_graph())

    def _init_adver_agents(self):
        agent = Agent(self.args)
        return agent

    def _init_good_agents(self):
        agents = []
        for i in range(self.args.num_adversaries, self.args.n_players):
            agent = AgentBaseline(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                thoughts = self.adver_agents.get_thoughts(s[:self.args.num_adversaries])  # tensor(nagents, actor_hidden_size)
                C = self.adver_agents.initiate_group(s[:self.args.num_adversaries], 3, thoughts)  # 这个数字4是进入通信的数量
            inter_thoughts = self.adver_agents.update_thoughts(thoughts, C)  # (nagents, actor_hidden_size)
            with torch.no_grad():  # 不需要计算梯度，也不会进行反向传播
                action_n = self.adver_agents.select_action2(thoughts, inter_thoughts, C)
                u = action_n
                actions = action_n
                for agent_id, agent in enumerate(self.good_agents):
                    action_good = agent.select_action(s[agent_id + self.args.num_adversaries], self.noise, self.epsilon)
                    actions.append(action_good)
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.num_adversaries], u[:self.args.num_adversaries], r[:self.args.num_adversaries], s_next[:self.args.num_adversaries],
                                      done[:self.args.num_adversaries],
                                      C.numpy().tolist())
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                self.adver_agents.learn(transitions)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns0, returns1, returns2, returns3 = self.evaluate()
                summary_str = sess.run(self.summary_ops, feed_dict={
                    self.summary_vars[0]: returns0,
                    self.summary_vars[1]: returns1,
                    self.summary_vars[2]: returns2,
                    self.summary_vars[3]: returns3,
                })
                self.writer.add_summary(summary_str, time_step + 0)
                self.writer.flush()
                self.returns.append(returns0)
                plt.figure()
                plt.plot(range(len(self.returns)), self.returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))  # 在之前的之上
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            np.save(self.save_path + '/returns.pkl', self.returns)

    def evaluate(self):
        returns0 = []
        returns1 = []
        returns2 = []
        returns3 = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            thoughts = self.adver_agents.get_thoughts(s[:self.args.num_adversaries])  # tensor(nagents, actor_hidden_size)
            C = self.adver_agents.initiate_group(s[:self.args.num_adversaries], 3, thoughts)
            rewards0 = 0
            rewards1 = 0
            rewards2 = 0
            rewards3 = 0
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                with torch.no_grad():
                    inter_thoughts = self.adver_agents.update_thoughts(thoughts, C)  # (nagents, actor_hidden_size)
                    action_n = self.adver_agents.select_action2(thoughts, inter_thoughts, C)
                    actions = action_n
                    for agent_id, agent in enumerate(self.good_agents):
                        action = agent.select_action(s[agent_id + self.args.num_adversaries], 0, 0)
                        actions.append(action)
                s_next, r, done, info = self.env.step(actions)
                rewards0 += r[0]
                rewards1 += r[1]
                rewards2 += r[2]
                rewards3 += r[3]
                s = s_next
            returns0.append(rewards0)
            returns1.append(rewards1)
            returns2.append(rewards2)
            returns3.append(rewards3)
            print('Returns is', rewards0)
        return sum(returns0) / self.args.evaluate_episodes, \
               sum(returns1) / self.args.evaluate_episodes, \
               sum(returns2) / self.args.evaluate_episodes, \
               sum(returns3) / self.args.evaluate_episodes


def build_summaries():
    episode_reward0 = tf.Variable(0., name="episode_reward0")
    tf.summary.scalar("Reward0", episode_reward0)
    episode_reward1 = tf.Variable(0., name="episode_reward1")
    tf.summary.scalar("Reward1", episode_reward1)
    episode_reward2 = tf.Variable(0., name="episode_reward2")
    tf.summary.scalar("Reward2", episode_reward2)
    episode_reward3 = tf.Variable(0., name="episode_reward3")
    tf.summary.scalar("Reward3", episode_reward3)

    summary_vars = [episode_reward0, episode_reward1, episode_reward2, episode_reward3]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
