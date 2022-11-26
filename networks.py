import torch
from torch import nn as nn
from torch.nn import functional as F


class CriticNetwork(nn.Module):
    def __init__(self, config):
        super(CriticNetwork, self).__init__()
        self.device = config['device']
        self.inputLayer = nn.Linear(config["state_size"], config["state_size"] * 2)
        self.layer1 = nn.Linear(config["state_size"] * 2, config["state_size"] * 2)
        self.outputLayer = nn.Linear(config["state_size"] * 2, 1)

    def forward(self, input):
        input = input.to(self.device)
        input = torch.tanh(self.inputLayer(input))
        input = torch.tanh(self.layer1(input))
        input = self.outputLayer(input)
        return input


class ActorNetwork(nn.Module):
    def __init__(self, config):
        super(ActorNetwork, self).__init__()
        self.device = config['device']
        self.inputLayer = nn.Linear(config["state_size"], config["state_size"] * 2)
        self.layer1 = nn.Linear(config["state_size"] * 2, config["state_size"] * 2)
        self.outputLayer = nn.Linear(config["state_size"] * 2, config["action_size"])

    def forward(self, input):
        input = input.to(self.device)
        input = torch.tanh(self.inputLayer(input))
        input = torch.tanh(self.layer1(input))
        input = F.softmax(self.outputLayer(input), dim=-1)
        return input


class CriticNetwork_v2(nn.Module):
    def __init__(self, config):
        super(CriticNetwork_v2, self).__init__()
        self.device = config['device']
        self.inputLayer = nn.Linear(config["state_size"], config["state_size"] * 2)
        self.layer1 = nn.Linear(config["state_size"] * 2, config["state_size"] * 2)
        self.outputLayer = nn.Linear(config["state_size"] * 2, 1)

    def forward(self, input):
        input = input.to(self.device)
        input = torch.relu(self.inputLayer(input))
        input = torch.relu(self.layer1(input))
        input = self.outputLayer(input)
        return input


class ActorNetwork_v2(nn.Module):
    def __init__(self, config):
        super(ActorNetwork_v2, self).__init__()
        self.device = config['device']
        self.inputLayer = nn.Linear(config["state_size"], config["state_size"] * 2)
        self.layer1 = nn.Linear(config["state_size"] * 2, config["state_size"] * 2)
        self.outputLayer = nn.Linear(config["state_size"] * 2, config["action_size"])

    def forward(self, input):
        input = input.to(self.device)
        input = torch.relu(self.inputLayer(input))
        input = torch.relu(self.layer1(input))
        input = F.softmax(self.outputLayer(input), dim=-1)
        return input



class CriticNetwork_v3(nn.Module):
    def __init__(self, config):
        super(CriticNetwork_v3, self).__init__()
        self.device = config['device']
        self.inputLayer = nn.Linear(config["state_size"], config["state_size"] * 3)
        self.layer1 = nn.Linear(config["state_size"] * 3, config["state_size"] * 3)
        self.outputLayer = nn.Linear(config["state_size"] * 3, 1)

    def forward(self, input):
        input = input.to(self.device)
        input = torch.relu(self.inputLayer(input))
        input = torch.relu(self.layer1(input))
        input = self.outputLayer(input)
        return input

class ActorNetwork_v3(nn.Module):
    def __init__(self, config):
        super(ActorNetwork_v3, self).__init__()
        self.device = config['device']
        self.inputLayer = nn.Linear(config["state_size"], config["state_size"] * 3)
        self.layer1 = nn.Linear(config["state_size"] * 3, config["state_size"] * 3)
        self.outputLayer = nn.Linear(config["state_size"] * 3, config["action_size"])

    def forward(self, input):
        input = input.to(self.device)
        input = torch.relu(self.inputLayer(input))
        input = torch.relu(self.layer1(input))
        input = F.softmax(self.outputLayer(input), dim=-1)
        return input