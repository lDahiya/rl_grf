import numpy as np
from matplotlib import pyplot as plt


class Plots:
    def __init__(self, config):
        self.envName = config['env_name']
        self.modelName = config['model_name']
        self.path = config["plots_path"]
        self.reward_ppo = list()
        self.average_reward_ppo = list()
        self.reward_test_ppo = list()

    def plot(self, data, title, lab):
        plt.plot(data)
        plt.title(title, loc='center', wrap=True)
        plt.xlabel(lab[0])
        plt.ylabel(lab[1])
        # plt.show()
        plt.savefig(self.path + title + '.png')
        plt.clf()

    def ppo_plots(self):
        axis_e = ["Episodes", "Epsilon"]
        axis_r = ["Episodes", "Total Rewards"]
        axis_r2 = ["Episodes", "Average Total Rewards"]
        if len(self.reward_ppo) > 0:
            self.plot(np.array(self.reward_ppo, dtype=float), self.envName + " Reward(TRAIN)" + "_" + self.modelName,
                      axis_r)
        if len(self.reward_test_ppo) > 0:
            self.plot(np.array(self.reward_test_ppo, dtype=float),
                      self.envName + " Reward(TEST)" + "_" + self.modelName,
                      axis_r)
        if len(self.average_reward_ppo) > 0:
            self.plot(np.array(self.average_reward_ppo, dtype=float),
                      self.envName + " Averaged Reward(TRAIN)" + "_" + self.modelName,
                      axis_r2)
