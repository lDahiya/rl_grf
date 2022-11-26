import getopt
import os
import random
import string
import sys

import gfootball.env as football_env
import torch

from helper import check_configuration, loadConfigurationForModel
from plotter import Plots
from ppo import Agent


def initializeEnvironment(config):
    print("=" * 10 + " Initializing Environment " + "=" * 10)
    env_name = "academy_single_goal_versus_lazy"
    env = football_env.create_environment(env_name=env_name, representation='simple115v2',
                                          render=False, rewards='scoring,checkpoints')
    config['state_size'] = env.observation_space.shape[0]
    config['action_size'] = env.action_space.n
    if not config['preload_model']:
        config['env_name'] = config['env_name'] + '_' + env_name
    return env


def initHelperPlotterEnvironment(config):
    env = initializeEnvironment(config)
    plotter = Plots(config)
    return env, plotter


def printConfig(configuration):
    print("=" * 10 + " Printing Configuration " + "=" * 10)
    print(configuration)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


def loadConfigurationAndTrain(configuration):
    testMode = False
    check_configuration(configuration)
    configuration = loadConfigurationForModel(configuration)
    env, plotter = initHelperPlotterEnvironment(configuration)
    printConfig(configuration)
    if not configuration["preload_model"]:
        print("=" * 10 + " Creating Fresh Model " + "=" * 10)
        agent = Agent(configuration)
        agent.trainWrapper(env, plotter)
    else:
        print("=" * 10 + " Loading Old Model " + "=" * 10)
        agent = Agent(configuration, preload=True)
        testMode = True
    return agent, env, plotter, testMode


def savePlotsAndModel(agent, plotter, config):
    if not config['preload_model']:
        print("=" * 10 + " Saving Model " + "=" * 10)
        agent.save()
    print("=" * 10 + " Generating and Saving plots " + "=" * 10)
    plotter.ppo_plots()
    return


def controller(agentName=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_ep_len = 10000
    if agentName is None:
        agentName = ''.join(random.choice(string.ascii_letters) for i in range(10))
    configuration = {
        'model_name': agentName,
        'env_name': "GRF",
        "gamma": 0.99,
        "learning_rate_actor": 0.0003,
        "learning_rate_critic": 0.001,
        "max_episode": max_ep_len,
        "max_test_episode": 50,
        "sync_time": max_ep_len * 4,
        "max_timestep": 3000000,
        "epochs": 40,
        "clip": 0.2,
        "action_size": 2,
        # "score": 470,
        "print": 100,
        "print_test": 10,
        "render_mode": True,
        "model_path": os.getcwd() + "\\models\\",
        "plots_path": os.getcwd() + "\\plots_ppo\\",
        "device": device,
        "actor_class": 'ActorNetwork_v3',
        "critic_class": 'CriticNetwork_v3'
    }
    agent, env, plotter, testMode = loadConfigurationAndTrain(configuration)
    agent.testWrapper(env, plotter, testMode)
    savePlotsAndModel(agent, plotter, configuration)
    env.close()


def process(argv):
    modelName = None
    opts, args = getopt.getopt(argv, "hi:o:", ["model=", "render="])
    for opt, arg in opts:
        if opt in ("-model", "--model"):
            modelName = arg
    controller(modelName)


if __name__ == '__main__':
    process(sys.argv[1:])
