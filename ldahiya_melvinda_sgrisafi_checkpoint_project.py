import gfootball.env as football_env
import copy
import os

import gym
import random
import numpy as np

import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class Plots:
    def __init__(self, config):
        self.epsilon_dqn = np.zeros(config["max_episode"])
        self.reward_dqn = np.zeros(config["max_episode"])
        self.average_reward_dqn = np.zeros(config["max_episode"])
        self.reward_test_dqn = np.zeros(config["max_test_episode"])
        self.epsilon_ddqn = np.zeros(config["max_episode"])
        self.reward_ddqn = np.zeros(config["max_episode"])
        self.average_reward_ddqn = np.zeros(config["max_episode"])
        self.reward_test_ddqn = np.zeros(config["max_test_episode"])

    def plot(self, data, title, lab):
        plt.plot(data)
        plt.title(title)
        plt.xlabel(lab[0])
        plt.ylabel(lab[1])
        plt.savefig(os.getcwd() + "\\plots\\" + title + '.png')
        plt.clf()

    def dplot(self, data1, data2, title, lab):
        plt.plot(data1, color='red', label=lab[0])
        plt.plot(data2, color='blue', label=lab[1])
        plt.title(title)
        plt.xlabel(lab[2])
        plt.ylabel(lab[3])
        plt.legend()
        plt.savefig(os.getcwd() + "\\plots\\" + title + '.png')
        plt.clf()

    def combined_plots(self, max_ep_dqn, max_ep_ddqn):
        axis_dr = ["DQN", "DDQN", "Episodes", "Total Rewards"]
        axis_dr2 = ["DQN", "DDQN", "Episodes", "Average Total Rewards"]
        self.dplot(self.reward_dqn[0:max_ep_dqn], self.reward_ddqn[0:max_ep_ddqn],
                   "DQN vs DDQN Algorithm Lunar Lander Environment(TRAIN)", axis_dr)
        self.dplot(self.average_reward_dqn[0:max_ep_dqn], self.average_reward_ddqn[0:max_ep_ddqn],
                   "DQN vs DDQN Algorithm Lunar Lander Environment(TRAIN)", axis_dr2)
        self.dplot(self.reward_test_dqn, self.reward_test_ddqn,
                   "DQN vs DDQN Algorithm Lunar Lander Environment(TEST)", axis_dr)

    def ddqn_plots(self, max_ep, environment_name):
        axis_e = ["Episodes", "Epsilon"]
        axis_r = ["Episodes", "Total Rewards"]
        axis_r2 = ["Episodes", "Average Total Rewards"]
        self.plot(self.epsilon_ddqn[0:max_ep], "DDQN " + environment_name + " Epsilon Decay", axis_e)
        self.plot(self.reward_ddqn[0:max_ep], "DDQN " + environment_name + " Reward(TRAIN)", axis_r)
        self.plot(self.reward_test_ddqn, "DDQN " + environment_name + " Reward(TEST)", axis_r)
        self.plot(self.average_reward_ddqn[0:max_ep], "DDQN " + environment_name + " Averaged Reward(TRAIN)", axis_r2)

    def dqn_plots(self, max_ep):
        axis_e = ["Episodes", "Epsilon"]
        axis_r = ["Episodes", "Total Rewards"]
        axis_r2 = ["Episodes", "Average Total Rewards"]
        self.plot(self.epsilon_dqn[0:max_ep], "DQN Lunar Lander Epsilon Decay", axis_e)
        self.plot(self.reward_dqn[0:max_ep], "DQN Lunar Lander Reward(TRAIN)", axis_r)
        self.plot(self.reward_test_dqn, "DQN Lunar Lander Reward(TEST)", axis_r)
        self.plot(self.average_reward_dqn[0:max_ep], "DQN Lunar Lander Averaged Reward(TRAIN)", axis_r2)


# Using Python Deque to enable FIFO structure
class ReplayBuffer:
    env_transition = namedtuple('env_transition', ("current_state", "action", "reward", "next_state", "terminate"))

    def __init__(self, config):
        self.buffer = deque([], config["buffer_capacity"])

    def sample(self, size):
        experiences = random.sample(self.buffer, size)
        states = torch.from_numpy(np.vstack([e.current_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.terminate for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def store(self, current_state, action, reward, next_state, terminate):
        self.buffer.append(self.env_transition(current_state, action, reward, next_state, terminate))

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inputLayer = nn.Linear(config["observation_size"] * 1, config["observation_size"] * 2)
        self.layer1 = nn.Linear(config["observation_size"] * 2, config["observation_size"] * 4)
        self.layer2 = nn.Linear(config["observation_size"] * 4, config["observation_size"] * 2)
        self.layer3 = nn.Linear(config["observation_size"] * 2, config["observation_size"] * 1)
        self.outputLayer = nn.Linear(config["observation_size"] * 1, config['action_size'])

    def forward(self, input):
        input = input.to(device)
        input = F.relu(self.inputLayer(input))
        input = F.relu(self.layer1(input))
        input = F.relu(self.layer2(input))
        input = F.relu(self.layer3(input))
        input = self.outputLayer(input)
        return input


class Agent:
    def __init__(self, config, environment, rbuffer):
        self.env = environment
        self.action_space = config["action_size"]
        self.start_epsilon = config["max_epsilon"]
        self.epsilon = self.start_epsilon
        self.end_epsilon = config["min_epsilon"]
        self.max_episode = config["max_episode"]
        self.gamma = config["gamma"]
        self.episodeNo = 0
        self.timestep = 0
        self.max_timestep = config["max_timestep"]
        self.print = config["print"]
        self.tau = config["tau"]
        self.sync = config["sync_time"]
        self.max_test_episode = config["max_test_episode"]
        self.sample_size = config["sample_size"]
        self.decay_factor = ((self.end_epsilon / self.start_epsilon) ** (1 / self.max_episode))
        self.rbuffer = rbuffer
        self.avnn = DQN(config).to(device)
        self.tnn = copy.deepcopy(self.avnn)
        self.tnn = DQN(config).to(device)
        self.score = config["score"]
        # self.opt = torch.optim.SGD(self.avnn.parameters(), lr=config["learning_rate"], momentum=0.9)
        self.opt = torch.optim.Adam(self.avnn.parameters(), lr=config["learning_rate"])
        self.criterion = nn.MSELoss()

    def policy(self, state):
        chance = random.uniform(0, 1)
        if chance <= self.epsilon:
            action = random.randint(0, self.action_space - 1)
        else:
            self.avnn.eval()
            with torch.no_grad():
                row = self.avnn(state)
            self.avnn.train()
            row = row.cpu()
            action = np.random.choice(np.flatnonzero(row == row.max()))
        return action

    def step(self):
        if len(self.rbuffer) > self.sample_size:
            states, actions, rewards, next_states, dones = self.rbuffer.sample(self.sample_size)
        return states, actions, rewards, next_states, dones

    def fill_buffer(self, env):
        state = torch.from_numpy(env.reset())
        while len(self.rbuffer) < self.rbuffer.buffer.maxlen:
            action = self.policy(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state)
            self.rbuffer.store(state, action, reward, next_state, done)
            if done:
                state = torch.from_numpy(env.reset())
            else:
                state = next_state

    def train_dqn(self, env, plotter):
        while self.episodeNo < self.max_episode:
            self.timestep = 0
            cumulativeScore = 0
            state = torch.from_numpy(env.reset())
            for t in range(0, self.max_timestep):
                action = self.policy(state)
                next_state, reward, done, info = env.step(action)
                next_state = torch.from_numpy(next_state)
                self.rbuffer.store(state, action, reward, next_state, done)
                if t % 4 == 0:
                    current_state, actions, rewards, next_states, dones = self.step()
                    predicted_targets = self.avnn(current_state).gather(1, actions)
                    with torch.no_grad():
                        labels_next = self.tnn(next_states).detach().max(1)[0].unsqueeze(1)
                    labels = rewards + (self.gamma * labels_next * (1 - dones))
                    self.opt.zero_grad()
                    loss = self.criterion(predicted_targets, labels).to(device)
                    loss.backward()
                    self.opt.step()
                state = next_state
                cumulativeScore += reward
                if done:
                    break
            if self.episodeNo % self.sync == 0:
                self.tnn.load_state_dict(self.avnn.state_dict())
            if self.episodeNo % self.print == 0:
                # print("Epsilon %s" % self.epsilon)
                print("Cumulative Reward for episode %s : %s" % (self.episodeNo, cumulativeScore))
            plotter.reward_dqn[self.episodeNo] = cumulativeScore
            plotter.epsilon_dqn[self.episodeNo] = self.epsilon
            av_sum = 0
            for x in range(0, self.max_test_episode):
                av_sum += plotter.reward_dqn[self.episodeNo - x]
            av_sum = av_sum / self.max_test_episode
            plotter.average_reward_dqn[self.episodeNo] = av_sum
            if av_sum >= self.score:
                self.max_episode = self.episodeNo
                break
            self.epsilon = self.epsilon * self.decay_factor
            self.episodeNo += 1

    def train_ddqn(self, env, plotter):
        average_t = 0
        while self.episodeNo < self.max_episode:
            self.timestep = 0
            cumulativeScore = 0
            state = torch.from_numpy(env.reset())
            for t in range(0, self.max_timestep):
                action = self.policy(state)
                next_state, reward, done, info = env.step(action)
                env.render()
                next_state = torch.from_numpy(next_state)
                self.rbuffer.store(state, action, reward, next_state, done)
                if t % 4 == 0:
                    current_state, actions, rewards, next_states, dones = self.step()
                    predicted_targets = self.avnn(current_state).gather(1, actions)
                    self.avnn.eval()
                    with torch.no_grad():
                        max_actions = self.avnn(next_states).detach().max(1)[1].unsqueeze(1).long()
                        labels_next = self.tnn(next_states).gather(1, max_actions)
                    self.avnn.train()
                    self.opt.zero_grad()
                    labels = rewards + (self.gamma * labels_next * (1 - dones))
                    loss = self.criterion(predicted_targets, labels).to(device)
                    loss.backward()
                    self.opt.step()
                state = next_state
                cumulativeScore += reward
                if done:
                    average_t += t
                    break
            if self.episodeNo % self.sync == 0:
                for target_p, policy_p in zip(self.tnn.parameters(), self.avnn.parameters()):
                    updatedData = (1 - self.tau) * target_p.data + self.tau * policy_p.data
                    target_p.data.copy_(updatedData)
            if self.episodeNo % self.print == 0:
                # print("Epsilon %s" % self.epsilon)
                print(
                    "Cumulative Reward for episode %s : %s\nAverage Timestep : %s" % (
                        self.episodeNo, cumulativeScore, (average_t / (self.episodeNo + 1))))
            plotter.reward_ddqn[self.episodeNo] = cumulativeScore
            plotter.epsilon_ddqn[self.episodeNo] = self.epsilon
            av_sum = 0
            for x in range(0, self.max_test_episode):
                av_sum += plotter.reward_ddqn[self.episodeNo - x]
            av_sum = av_sum / self.max_test_episode
            plotter.average_reward_ddqn[self.episodeNo] = av_sum
            if av_sum >= self.score:
                self.max_episode = self.episodeNo
                break
            self.epsilon = self.epsilon * self.decay_factor
            self.episodeNo += 1

    def test(self, env, plotArray, config):
        for i in range(config['max_test_episode']):
            state = torch.from_numpy(env.reset())
            total_reward = 0
            for j in range(500):
                self.avnn.eval()
                with torch.no_grad():
                    row = self.avnn(state)
                row = row.cpu()
                action = np.random.choice(np.flatnonzero(row == row.max()))
                state, reward, done, info = env.step(action)
                total_reward += reward
                state = torch.from_numpy(state)
                if config["render_testing"] and i % config["render_testing_rate"]:
                    env.render()
                if done:
                    print("Finished testing %s with reward %s" % (i, total_reward))
                    break
            plotArray[i] = total_reward

        env.close()
        return


# def process_dqn(configuration, plotter):
#     print(configuration)
#     print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     rbuffer = ReplayBuffer(configuration)
#     agent = Agent(configuration, env, rbuffer)
#     agent.fill_buffer(env)
#     agent.train_dqn(env, plotter)
#     # env = gym.make("LunarLander-v2", render_mode="human")
#     agent.test(env, plotter.reward_test_dqn, configuration)
#     env.close()
#     return agent.max_episode


def process_ddqn(env, configuration, plotter):
    print("=" * 10 + " Loading Configuration " + "=" * 10)
    print(configuration)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    rbuffer = ReplayBuffer(configuration)
    agent = Agent(configuration, env, rbuffer)
    print("=" * 10 + " Filling Buffer " + "=" * 10)
    agent.fill_buffer(env)
    print("=" * 10 + " Beginning Training " + "=" * 10)
    agent.train_ddqn(env, plotter)
    # ftball_env = football_env.create_environment(env_name='academy_empty_goal_close', representation='simple115v2',
    #                                              render=True)
    print("=" * 10 + " Beginning Testing " + "=" * 10)
    agent.test(env, plotter.reward_test_ddqn, configuration)
    print("=" * 10 + " Cleaning up " + "=" * 10)
    env.close()
    return agent.max_episode


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configuration = {
    "buffer_capacity": 20000,
    "gamma": 0.99,
    "learning_rate": 0.0001,
    "max_episode": 100000,
    "max_test_episode": 30,
    "max_epsilon": 1,
    "min_epsilon": 0.001,
    "sample_size": 128,
    "sync_time": 2,
    "max_timestep": 5000,
    "score": 0.95,
    "print": 500,
    "tau": 0.08,
    "render_testing": True,
    "render_testing_rate": 10

}


def loadDynamicConfiguration(env, config):
    config['action_size'] = env.action_space.n
    config['observation_size'] = env.observation_space.shape[0]
    return config


env_array = [
    # "academy_pass_and_shoot_with_keeper",
    #          "academy_run_pass_and_shoot_with_keeper",
    #          "academy_3_vs_1_with_keeper",
    #          "academy_corner",
    #          "academy_counterattack_easy",
    "academy_single_goal_versus_lazy"
]

for x in env_array:
    configuration["environment_name"] = x
    ftball_env = football_env.create_environment(env_name=x, representation='simple115v2',
                                                 render=False)
    configuration = loadDynamicConfiguration(ftball_env, configuration)

    plotter = Plots(configuration)
    ddqn_max_episode = process_ddqn(ftball_env, configuration, plotter=plotter)
    print("=" * 10 + " Generating Plots " + "=" * 10)
    plotter.ddqn_plots(ddqn_max_episode, x)
