import json
from pydoc import locate

import torch
import torch.nn as nn
from torch.distributions import Categorical

from helper import loadConfigurationFile
from networks import CriticNetwork, ActorNetwork


class rBuffer:
    def __init__(self, config):
        self.states = []
        self.actions = []
        self.actionProbs_log = []
        self.rewards = []
        self.dones = []
        self.device = config['device']

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.actionProbs_log.clear()
        self.rewards.clear()
        self.dones.clear()

    def sample(self):
        states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        actionProbs_log = torch.squeeze(torch.stack(self.actionProbs_log, dim=0)).detach().to(self.device)
        return states, actions, actionProbs_log

    def getCumulativeRewards(self, gamma):
        cumulativeRewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            cumulativeRewards.insert(0, discounted_reward)
        cumulativeRewards = torch.tensor(cumulativeRewards, dtype=torch.float32).to(self.device)
        cumulativeRewards = (cumulativeRewards - cumulativeRewards.mean()) / (cumulativeRewards.std() + 1e-6)
        return cumulativeRewards


class Agent:
    def __init__(self, config, preload=False):
        self.config = config
        self.device = config['device']
        self.gamma = config['gamma']
        self.clip = config['clip']
        self.epochs = config['epochs']
        self.rbuffer = rBuffer(config)
        self.max_episode = config["max_episode"]
        self.max_test_episode = config["max_test_episode"]
        actor, critic, actor_old = self.initilizeNetwork(config, preload)
        self.actor = actor
        self.critic = critic
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config["learning_rate_actor"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config["learning_rate_critic"])
        self.actor_old = actor_old
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.criterion = nn.MSELoss()

    def initilizeNetwork(self, config, preload):
        ActorClass = locate('networks.' + config['actor_class'])
        CriticClass = locate('networks.' + config['critic_class'])
        if not preload:
            actor = ActorClass(config).to(config['device'])
            critic = CriticClass(config).to(config['device'])
            actor_old = ActorClass(config).to(config['device'])
        else:
            actor = ActorClass(config)
            actor.load_state_dict(torch.load(self.config['model_path_actor']))
            actor.eval()
            actor.to(self.device)
            actor_old = ActorClass(config).to(config['device'])

            critic = CriticClass(config)
            critic.load_state_dict(torch.load(self.config['model_path_critic']))
            critic.eval()
            critic.to(self.device)
        return actor, critic, actor_old

    def policy(self, state):
        actionProbs = self.actor(state)
        dist = Categorical(actionProbs)
        action = dist.sample()
        actionProbs_log = dist.log_prob(action)
        return action.detach(), actionProbs_log.detach()

    def evaluate(self, state, action):
        actionProbs = self.actor(state)
        dist = Categorical(actionProbs)
        actionProbs_log = dist.log_prob(action)
        vf = self.critic(state)
        return actionProbs_log, torch.squeeze(vf)

    def getAction(self, state):
        with torch.no_grad():
            action, actionProb_logs = self.policy(state)
        self.rbuffer.states.append(state)
        self.rbuffer.actions.append(action)
        self.rbuffer.actionProbs_log.append(actionProb_logs)
        return action.item()

    def step(self):
        cumulativeRewards = self.rbuffer.getCumulativeRewards(self.gamma)
        prev_states, prev_actions, prev_actionProbs_log = self.rbuffer.sample()
        for x in range(self.epochs):
            actionProbs_log, vf = self.evaluate(prev_states, prev_actions)
            policy_ratio = torch.exp(actionProbs_log - prev_actionProbs_log.detach())
            adv = cumulativeRewards - vf.detach()
            loss1 = policy_ratio * adv
            loss2 = torch.clamp(policy_ratio, 1 - self.clip, 1 + self.clip) * adv
            act_loss = (-1 * torch.min(loss1, loss2)).mean()
            crit_loss = self.criterion(vf, cumulativeRewards)
            self.actor_opt.zero_grad()
            act_loss.backward(retain_graph=True)
            self.actor_opt.step()
            self.critic_opt.zero_grad()
            crit_loss.backward()
            self.critic_opt.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.rbuffer.clear()

    def save(self):
        del self.config["device"]
        self.config["preload_model"] = True
        data = loadConfigurationFile(self.config["model_path"])
        data[self.config["model_name"]] = self.config
        json_object = json.dumps(data, indent=4)
        with open(self.config["model_path"] + "configurations.json", "w") as outfile:
            outfile.write(json_object)
        torch.save(self.actor.state_dict(), self.config['model_path_actor'])
        torch.save(self.critic.state_dict(), self.config['model_path_critic'])

    def test(self, env, plotter, testMode):
        for episodeNo in range(1, self.config['max_test_episode'] + 1):
            state = torch.from_numpy(env.reset())
            total_reward = 0
            for t in range(1, self.config['max_episode'] + 1):
                self.actor.eval()
                action = self.getAction(state)
                next_state, reward, done, info = env.step(action)
                state = torch.from_numpy(next_state)
                total_reward += reward
                # if testMode and ((episodeNo + 1) % self.config['max_test_episode'] == 0) and self.config['render_mode']:
                env.render()
                if done:
                    if episodeNo % self.config['print_test'] == 0:
                        print("Finished testing %s with reward %s" % (episodeNo, total_reward))
                    break
            self.rbuffer.clear()
            plotter.reward_test_ppo.append(total_reward)

    def train(self, env, plotter):
        ts = 0
        episodeNo = 1
        while ts <= self.config['max_timestep']:
            state = torch.from_numpy(env.reset())
            current_ep_reward = 0
            for t in range(1, self.config['max_episode'] + 1):
                action = self.getAction(state)
                next_state, reward, done, info = env.step(action)
                state = torch.from_numpy(next_state)
                self.rbuffer.rewards.append(reward)
                self.rbuffer.dones.append(done)
                ts += 1
                current_ep_reward += reward
                if ts % self.config['sync_time'] == 0:
                    self.step()
                if done:
                    break
            plotter.reward_ppo.append(current_ep_reward)
            av_sum = sum(plotter.reward_ppo[-self.max_test_episode:])
            av_sum = av_sum / self.max_test_episode
            plotter.average_reward_ppo.append(av_sum)
            if episodeNo % self.config['print'] == 0:
                print("Cumulative Reward and Average Reward for episode %s : %s %s" % (
                    episodeNo, current_ep_reward, plotter.average_reward_ppo[episodeNo - 1]))
            episodeNo += 1
        return

    def trainWrapper(self, env, plotter):
        print("=" * 10 + " Training Agent " + "=" * 10)
        self.train(env, plotter)
        return

    def testWrapper(self, env, plotter, testMode):
        print("=" * 10 + " Testing Agent " + "=" * 10)
        self.test(env, plotter, testMode)
        return
