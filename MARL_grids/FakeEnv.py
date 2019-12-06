import numpy as np
## With contrib to use sift
import cv2
import imagehash
from PIL import Image
from collections import defaultdict

IMG_H, IMG_W = 856, 480

class FakeEnv(object):
    def __init__(self, grid_size, max_visit=5):
        self.grid_size = grid_size
        self.position = np.zeros(2, dtype=int)
        self.visited = np.zeros(grid_size)
        ## By grids
        self.ind2action = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }
        self.imgstack = set()
        self.features = np.zeros(grid_size, dtype=int)
        self.last_position = self.position

        self.max_visit = max_visit
        ## This defines how many times a grid should be visited
        self.proper_visit = np.zeros(grid_size, dtype=int)

        self.fake_keypnts = np.asarray([[10, 200000, 500000],
                                        [250, 100000, 35000],
                                        [50000, 7000, 190000]])

    def get_state(self):
        ## Encode the features into state
        try:
            feature_state = self.features[int(self.position[0])]
        except:
            feature_state = 0
        return self.position.tolist()

    def reset(self):
        self.position = np.zeros(2)
        # self.last_position = self.position
        self.visited = np.zeros((self.grid_size[0], self.grid_size[1]))
        self.visited[0, 0] += 1
        self.features = np.zeros(self.grid_size)
        # self.visited = np.zeros((self.grid_size[0] + 2, self.grid_size[1] + 2), dtype=int)
        return self.get_state()

    def compute_reward(self):
        visited_times = self.visited[int(self.position[0]), int(self.position[1])]
        if visited_times == 0:
            return 5
        elif visited_times <= self.max_visit:
            return 1
        elif visited_times > self.max_visit:
            return -5

    def compute_reward_imghash(self):
        pass
    
    def compute_reward_img_feature(self):
        y = int(self.position[0])
        z = int(self.position[1])
        visited_times = self.visited[y, z]
        if visited_times == 0:
            return 2
        elif visited_times <= self.proper_visit:
            ## TELL THE AGENT THAT IT IS OK TO STAY IN THE FEATURE-TICH REGION A LITTLE BIT LONGER
            # len_keypnts = self.fake_keypnts[y, z] + int(np.floor(self.fake_keypnts[y, z] * (np.random.random() - 0.3)))
            # len_keypnts = len_keypnts // 856
            # ## Calculate the imcremental mean of the features
            # last_mu = self.features[y, z]
            # n = self.visited[y, z]
            # mu = (len_keypnts + n * last_mu - last_mu) / n // 240
            # # mu = len_keypnts // 500
            # self.features[y, z] = mu
            # print(.5 + mu)
            # return .5 + mu
            print(self.fake_keypnts[y, z] / 500000)
            return self.fake_keypnts[y, z] / 500000
        elif visited_times > self.proper_visit:
            return -1

    # def circular_move(self, action):
    #     movement = self.ind2action[action]
    #     next_y = self.position[0] + movement[0]
    #     next_z = self.position[1] + movement[1]


    def step(self, action, img):
        movement = self.ind2action[action]
        next_y = self.position[0] + movement[0]
        next_z = self.position[1] + movement[1]
        if next_y < 0 or next_y >= self.grid_size[0] or next_z < 0 or next_z >= self.grid_size[1]:
            # self.last_position = self.position
            self.position = self.position + np.asarray(movement)
            return self.get_state(), -50, True
        # self.last_position = self.position
        '''Add hash value to the stack'''
        imghash = imagehash.average_hash(Image.fromarray(img))
        self.imgstack.add(imghash)

        self.position = self.position + np.asarray(movement)
        self.visited[int(self.position[0]), int(self.position[1])] += 1
        reward = self.compute_reward()
        # reward = self.compute_reward_img_feature()
        done = False
        if (np.min(self.visited) >= self.max_visit):
            done = True
        if (np.max(self.visited) >= 10):
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
            self.Q = np.zeros((grid_size[0], grid_size[1], n_actions))
        else:
            self.Q = np.load(Q_path)

    def save_Q(self):
        np.save('Q_table_feature_reward.npy', self.Q)

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
            rand = np.random.random()
            if rand >= epsilon:
                return np.argmax(self.Q[int(observation[0]), int(observation[1]), :])
            return np.random.choice(self.n_actions)

        return policy_fn

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
            # Print out which episode we're on, useful for debugging.
            # if i_episode % 1000 == 0:
            #     print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")

            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            episode = []
            state = env.reset()
            # print('begin episodes')
            for t in range(100):
                action = policy(state)
                next_state, reward, done = env.step(action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state
            # print('episodes done')
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
                self.Q[int(state[0]), int(state[1]), action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        self.save_Q()
            # The policy is improved implicitly by changing the Q dictionary
        
    
def test(agent, env):
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

if __name__ == '__main__':
    img_save = './imgs2'
    grid_len = .3
    grid_size = (3, 3)
    num_episodes = 10000
    n_actions = 4

    # env = Environment(img_save, grid_len, grid_size)
    env = FakeEnv(grid_size)
    agent = Agent(grid_size, n_actions)
    # agent = Agent(grid_size, n_actions, './Q_table.npy')
    agent.optimize_model(env, num_episodes)
    print(env.features)
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
    
