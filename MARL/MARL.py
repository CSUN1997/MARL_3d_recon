import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from DroneManagement import Environment, ImgDatabase

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Agent:
    def __init__(self, n_actions, policy, target, BATCH_SIZE=128, GAMMA=0.999,\
        EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, TARGET_UPDATE=10):
        self.n_actions = n_actions
        self.policy = policy
        self.target = target
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE

        self.steps_done = 0

    # def get_state(self):
    #     ## position of the drone and the current image
    #     states = []
    #     for drone in self.drone_names:
    #         states.append((self.client.get_img(drone), self.client.get_location(drone)))
    #     return states

    ## epsilon greedy
    def select_action(self, state):
        sample = np.random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.Tensor([[np.random.randrange(self.n_actions)]], dtype=torch.Long)

    # def get_reward(self, states, actions, next_states):
    #     '''
    #     Reward does not have to be derivable, because rewards are smapled from environment, the environment is unknown.
    #     Make it condition on current state s, action a and next state s'.
    #     '''
    #     assert(len(states) == len(actions))
    #     assert(len(states) == len(next_states))
    #     reward = 0
    #     ## designed for multi-agent thus there will be multiple states and actions for individual drone
    #     for (cur_img, cur_location), action, (next_img, next_location) in zip(states, actions, next_states):
    #         reward += np.abs(self.imgDB.difference(cur_img) - self.imgDB.difference(next_img))
    #         ## add a reward term about location

    #     return reward


    def optimize_model(self, memory, optimizer):
        if len(memory) < self.BATCH_SIZE:
            return
        transitions = memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

if __name__ == '__main__':
    num_episodes = 50
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        drone = 'drone0'
        env = Environment(drone)
        memory = ReplayMemory(10000)
        # last_screen = get_screen()
        # current_screen = get_screen()
        # state = current_screen - last_screen
        ## output image sie
        h, w = 256, 256
        n_actions = 6
        imgDB = ImgDatabase('average')
        ## 
        policy = DQN(h, w, n_actions)
        target = DQN(h, w, n_actions)
        target.load_state_dict(policy.state_dict())
        target.eval()

        agent = Agent(n_actions, policy, target)

        ## inti state
        state = env.cur_state()
        imgDB.insert(state)

        lr = 0.00001
        optimizer = optim.Adam(policy.parameters(), lr)

        episode_durations = []
        
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            next_state, done = env.step(action.item())

            reward = imgDB.difference(next_state) / len(imgDB)
            reward = torch.Tensor([reward])

            # # Observe new state
            # last_screen = current_screen
            # current_screen = get_screen()
            # if not done:
            #     next_state = current_screen - last_screen
            # else:
            #     next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_model(memory, optimizer)
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % agent.TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())