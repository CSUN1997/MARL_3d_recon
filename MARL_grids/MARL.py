import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

from DroneManagement import Environment

class Agent:
    def __init__(self, grid_size, n_actions):
        self.n_actions = n_actions
        self.Q = np.zeros((grid_size[0], grid_size[1], n_actions))
        self.Q_saved =False

    def __del__(self):
        if not self.Q_saved:
            self.save_Q()

    def save_Q(self):
        np.save('./Q_table.npy', self.Q)
        self.Q_saved = True

    def transform(self, img):
        ## transform h, w ,c to c, h, w
        transformed = np.zeros((1, img.shape[2], img.shape[0], img.shape[1]))
        transformed[:, 0, :, :] = img[:, :, 0]
        transformed[:, 1, :, :] = img[:, :, 1]
        transformed[:, 2, :, :] = img[:, :, 2]
        return transformed

    def make_epsilon_greedy_policy(self, epsilon):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        
        Args:
            epsilon: The probability to select a random action. Float between 0 and 1.
        
        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        
        """
        # def policy_fn(observation):
        #     A = np.ones(self.n_actions, dtype=float) * epsilon / self.n_actions
        #     best_action = np.argmax(self.Q[observation[0], observation[1]])
        #     A[best_action] += (1.0 - epsilon)
        #     return A

        def policy_fn(ob0.servation):
            rand = np.random.random()
            if rand >= epsilon:
                return np.argmax(self.Q[observation[0], observation[1]])
            return np.random.choice(self.n_actions)

        return policy_fn

    def optimize_model(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        policy = self.make_epsilon_greedy_policy(epsilon)
        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                # sys.stdout.flush()
            
            # Reset the environment and pick the first action
            state = env.reset()
            
            # One step in the environment
            # total_reward = 0.0
            for t in count():
                # Take a step
                action = policy(state)
                # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done = env.step(action)
                
                # TD Update
                best_next_action = np.argmax(self.Q[next_state[0], next_state[1]])    
                td_target = reward + discount_factor * self.Q[next_state[0], next_state[1]][best_next_action]
                td_delta = td_target - self.Q[state[next_state[0], next_state[1]]][action]
                self.Q[state[0], state[1]][action] += alpha * td_delta
                if done:
                    break
                    
                state = next_state
    

if __name__ == '__main__':
    img_save = './imgs'
    drone_name = 'bebop'
    grid_len = 30.
    grid_size = (3, 3)
    num_episodes = 10
    n_actions = 4

    env = Environment(img_save, drone_name, grid_len, grid_size, 10)
    agent = Agent(grid_size, n_actions)
    agent.optimize_model(env, num_episodes)
    
