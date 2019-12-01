import numpy as np
import matplotlib.pyplot as plt
from itertools import count

from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from DroneManagement_grid import FakeEnv
from collections import defaultdict
import imagehash
import cv2
from PIL import Image
import time


class UserVision:
    '''
    For taking pictures. By default, images keep streaming in and there will be 30 images per sec. The use of imagehash makes
    sure that the drone won't take similar images.
    '''
    def __init__(self, vision):
        self.index = 0
        self.vision = vision
        self.last_hash = None
        self.threshold = 3

    def save_pictures(self, args):
        # print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            imghash = imagehash.average_hash(Image.fromarray(img))
            if (self.last_hash is not None) and (np.abs(imghash - self.last_hash) <= self.threshold):
                return
            filename = "imgs2/test_image_%06d.jpg" % self.index
            # uncomment this if you want to write out images every time you get a new one
            cv2.imwrite(filename, img)
            img.dtype = np.uint8
            self.last_hash = imghash
            self.index +=1


class Environment(object):
    def __init__(self, img_save, grid_len, grid_size, proper_visit=1):
        self.bebop = Bebop()
        self.bebop.connect(10)
        ## To take pictures
        self.droneVision = DroneVision(self.bebop, is_bebop=True, buffer_size=200)
        # self.droneVision.open_video()
        self.userVision = UserVision(self.droneVision)
        self.droneVision.set_user_callback_function(self.userVision.save_pictures, user_callback_args=None)
        self.droneVision.open_video()

        self.grid_len = grid_len
        self.grid_size = grid_size
        self.position = np.zeros(2)
        self.last_position = self.position
        self.visited = np.zeros((grid_size[0], grid_size[1]), dtype=int)
        ## By grids
        self.ind2action = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }

        ## This defines how many times a grid should be visited
        self.proper_visit = proper_visit
        print('Take off')
        self.bebop.safe_takeoff(10)
        ## zPositive is DOWN, y+ is right
        self.bebop.move_relative(0, 0, -1.3, 0)

    def __del__(self):
        self.bebop.move_relative(0, -self.position[0], 0, 0)
        self.delete()

    def delete(self):
        self.bebop.safe_land(10)
        self.bebop.stop_video_stream()
        self.bebop.disconnect()

    def reset(self):
        print('resetting by {}, {}'.format(-self.position[0], -self.position[1]))
        self.bebop.move_relative(0, -self.position[0], 0, 0)
        self.bebop.move_relative(0, 0, -self.position[1], 0)
        self.position = np.zeros(2)
        self.last_position = self.position
        self.visited = np.zeros(self.grid_size)
        self.visited[0, 0] += 1
        return self.get_state()

    def get_state(self):
        return self.last_position.tolist() + self.position.tolist()


    def compute_reward(self):
        # if np.any(self.visited == 0):
        #     return -np.sum(self.visited == 0)
        # distance = np.abs(self.proper_visit - self.visited)
        # ## Avoid division by 0
        # return np.sum(1 / np.maximum(distance, np.ones(self.visited.shape) * 1e-5))
        visited_times = self.visited[int(self.position[0]), int(self.position[1])]
        if visited_times == 0:
            return 20
        elif visited_times <= self.proper_visit:
            return 3
        elif visited_times > self.proper_visit:
            return -10

    def compute_reward_img_feature(self):
        pass

    def emergency(self):
        self.delete()

    def step(self, action):
        movement = self.ind2action[action]
        next_y = self.position[0] + movement[0]
        next_z = self.position[1] + movement[1]
        print(next_y, next_z)
        if next_y < 0 or next_y >= self.grid_size[0] or next_z < 0 or next_z >= self.grid_size[1]:
            self.last_position = self.position
            self.position = self.position + np.asarray(movement)
            return self.get_state(), -20, True
        self.bebop.move_relative(0, movement[0] * self.grid_len, movement[1] * self.grid_len, 0)
        self.last_position = self.position
        self.position = self.position + np.asarray(movement)
        
        self.visited[int(self.position[0]), int(self.position[1])] += 1
        reward = self.compute_reward()
        done = False
        if (np.min(self.visited) >= self.proper_visit) or (np.max(self.visited) >= 8):
            reward = -10
            done = True
        ## Calculate reward. If out of grid, assign a negative reward
        return self.get_state(), reward, done


class Agent:
    def __init__(self, grid_size, n_actions, Q_path=''):
        self.action_count = np.zeros(n_actions)
        self.n_actions = n_actions
        # self.Q = np.random.random((grid_size[0], grid_size[1], n_actions)) * 0.001
        if Q_path == '':
            self.Q = np.zeros((grid_size[0], grid_size[1], grid_size[0], grid_size[1], n_actions))
        else:
            self.Q = np.load(Q_path)

    def save_Q(self):
        np.save('Q_table.npy', self.Q)

    def make_epsilon_greedy_policy(self, epsilon):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        
        Args:
            epsilon: The probability to select a random action. Float between 0 and 1.
        
        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        
        """
        def policy_fn(observation):
            print(observation)
            rand = np.random.random()
            if rand >= epsilon:
                return np.argmax(self.Q[int(observation[0]), int(observation[1]), int(observation[2]), int(observation[3]), :])
            return np.random.choice(self.n_actions)

        return policy_fn


    # def optimize_model(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0):
    #     policy = self.make_epsilon_greedy_policy(epsilon)
    #     for i_episode in range(num_episodes):
    #         # Print out which episode we're on, useful for debugging.
    #         if (i_episode + 1) % 100 == 0:
    #             print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
    #             # sys.stdout.flush()
            
    #         # Reset the environment and pick the first action
    #         state = env.reset()
            
    #         # One step in the environment
    #         # total_reward = 0.0
    #         for t in count():
    #             # Take a step
    #             action = policy(state)
    #             self.action_count[int(action)] += 1
    #             # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    #             next_state, reward, done = env.step(action)
    #             if done:
    #                 break
    #             # TD Update
    #             best_next_action = np.argmax(self.Q[int(next_state[0]), int(next_state[1]), :])
    #             td_target = reward + discount_factor * self.Q[int(next_state[0]), int(next_state[1]), best_next_action]
    #             td_delta = td_target - self.Q[int(state[0]), int(next_state[1]), action]
    #             self.Q[int(state[0]), int(state[1]), action] += alpha * td_delta
                
    #             state = next_state

    #     self.save_Q()

    def optimize_model(self, env, num_episodes, discount_factor=1.0, epsilon=0.3):
        """
        Monte Carlo Control using Epsilon-Greedy policies.
        Finds an optimal epsilon-greedy policy.
        
        Args:
            env: OpenAI gym environment.
            num_episodes: Number of episodes to sample.
            discount_factor: Gamma discount factor.
            epsilon: Chance the sample a random action. Float betwen 0 and 1.
        
        Returns:
            A tuple (Q, policy).
            Q is a dictionary mapping state -> action values.
            policy is a function that takes an observation as an argument and returns
            action probabilities
        """
        
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        
        # The policy we're following
        policy = self.make_epsilon_greedy_policy(epsilon)
        
        for i_episode in range(1, num_episodes + 1):
            print(str(i_episode) + ' episode')
            # Print out which episode we're on, useful for debugging.
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")

            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            episode = []
            state = env.reset()
            print('begin episodes')
            for t in range(100):
                action = policy(state)
                next_state, reward, done = env.step(action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state
            print('episodes done')
            # Find all (state, action) pairs we've visited in this episode
            # We convert each state to a tuple so that we can use it as a dict key
            sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
            for state, action in sa_in_episode:
                sa_pair = (state, action)
                # Find the first occurance of the (state, action) pair in the episode
                first_occurence_idx = next((i for i, x in enumerate(episode) if x[0] == state and x[1] == action), None)
                # Sum up all rewards since the first occurance
                G = sum([x[2]*(discount_factor**i) for i, x in enumerate(episode[first_occurence_idx:])])
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                # Calculate average return for this state over all sampled episodes
                self.Q[int(state[0]), int(state[1]), int(state[2]), int(state[3]), action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        self.save_Q()
            # The policy is improved implicitly by changing the Q dictionary
        
    
def test(agent, env):
    # try:
    policy = agent.make_epsilon_greedy_policy(0)
    state = env.reset()
    print('===============================================')
    print(env.visited)
    done = False
    while not done:
        action = policy(state)
        next_state, reward, done = env.step(action)
        state = next_state
    print(env.visited)
    env.emergency()
    # except Exception as e:
    #     print(e)
    #     env.emergency()


if __name__ == '__main__':
    img_save = './imgs2'
    grid_len = .5
    grid_size = (3, 3)
    num_episodes = 500
    n_actions = 4

    env = Environment(img_save, grid_len, grid_size)
    # env = FakeEnv(grid_size)
    agent = Agent(grid_size, n_actions, './Q_table.npy')
    # agent.optimize_model(env, num_episodes)
    # print(env.visited)
    # print(agent.Q.shape)

    test(agent, env)
    # try:
    #     agent.optimize_model(env, num_episodes)
    # except:
    #     env.emergency()
    # print('================================')
    # print(agent.action_count)
    # print('================================')
    # print(agent.Q)
    
