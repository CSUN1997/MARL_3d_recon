import math
import random
import numpy as np
import matplotlib.pyplot as plt
import collections
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.categorical import Categorical

## Import environment built on the simulator
from SimulatorEnv import Environment

'''
This program aims at use neural networks to model policy. To make policy network more stable, baseline is added.
'''

class Policy(nn.Module):
    def __init__(self, n_state, n_action):
        super(Policy, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        
        self.policy_net = nn.Sequential(
            nn.Linear(n_state, 64),
            nn.Linear(64, n_action),
            nn.Softmax()
        )

    def forward(self, x):
        return self.policy_net(x)


class Value(nn.Module):
    def __init__(self, n_state):
        super(Value, self).__init__()
        self.n_state = n_state
        self.value_net = nn.Sequential(
            nn.Linear(n_state, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.value_net(x)


def select_action(policy_est, state, n_act):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state_ = torch.from_numpy(state)
    action_probs = policy_est(state_)
    return action_probs

def train(env, policy_est, value_est, optimizer_policy, optimizer_value, n_act, discount_factor=1, n_episode=10, lr=1e-4):
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    for i in range(n_episode):
        if (i + 1) % 10 == 0:
            print('{}/{} Episode:'.format(i + 1, n_episode))
        episode = []
        state = env.reset()
        ## Sample an episode
        for t in count():
            if (t + 1) % 10 == 0:
                print('\tSampling episode: {}th step'.format(t))
            action_probs = select_action(policy_est, state, n_act)
            action = np.random.choice(n_act, p=action_probs)
            next_state, reward, done = env.step(action)
            episode.append(Transition(state=state, action=action_probs[action], reward=reward, next_state=next_state, done=done))
            if done:
                break
            state = next_state

        ## Update the policy and value
        for t, transition in enumerate(episode):
            if (t + 1) % 10 == 0:
                print('\tUpdating policy and baseline: {}th step'.format(t))
            total_return = sum(discount_factor ** i * trans.reward for i, trans in enumerate(episode[t:]))
            baseline = value_est(torch.FloatTensor(transition.state))
            advantage = torch.FloatTensor(total_return) - baseline

            value_loss = nn.MSELoss(baseline, total_return)
            policy_loss = -torch.log(transition.action) * advantage
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer_policy.step()
            optimizer_value.step()


if __name__ == '__main__':
    grid_len = 0.5
    grid_size = (3, 3)
    n_state = 6
    n_action = 4
    lr = 1e-5

    env = Environment(grid_size, grid_len)
    policy_est = Policy(n_state, n_action)
    value_est = Value(n_state)

    optimizer_policy = optim.Adam(policy_est.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_est.parameters(), lr=lr)

    train(env, policy_est, value_est, optimizer_policy, optimizer_value, n_action)

