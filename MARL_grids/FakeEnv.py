import numpy as np
## With contrib to use sift
import cv2

class FakeEnv(object):
    def __init__(self, grid_size, proper_visit=5):
        self.grid_size = grid_size
        self.position = np.zeros(2, dtype=int)
        self.visited = np.zeros(grid_size, dtype=int)
        ## By grids
        self.ind2action = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }
        self.features = np.zeros(grid_size, dtype=int)
        self.last_position = self.position

        ## This defines how many times a grid should be visited
        self.proper_visit = proper_visit

        self.fake_keypnts = np.asarray([[10, 20, 50],
                                        [500, 100, 300],
                                        [500, 700, 900]])

    def get_state(self):
        return self.last_position.tolist() + self.position.tolist()

    def reset(self):
        self.position = np.zeros(2)
        self.last_position = self.position
        self.visited = np.zeros((self.grid_size[0], self.grid_size[1]))
        self.visited[0, 0] += 1
        # self.visited = np.zeros((self.grid_size[0] + 2, self.grid_size[1] + 2), dtype=int)
        return self.get_state()

    def compute_reward(self):
        visited_times = self.visited[int(self.position[0]), int(self.position[1])]
        if visited_times == 0:
            return 20
        elif visited_times <= self.proper_visit:
            return 3
        elif visited_times > self.proper_visit:
            return -10
    
    def compute_reward_img_feature(self):
        y = int(self.position[0])
        z = int(self.position[1])
        len_keypnts = self.fake_keypnts[y, z]
        ## Calculate the imcremental mean of the features
        last_mu = self.features[y, z]
        n = self.visited[y, z]
        mu = (len_keypnts + n * last_mu - last_mu) / n
        self.features[y, z] = mu
        print(last_mu, mu)
        if (mu - last_mu) <= 10:
            return -20
        else:
            return 20

    def step(self, action):
        movement = self.ind2action[action]
        next_y = self.position[0] + movement[0]
        next_z = self.position[1] + movement[1]
        if next_y < 0 or next_y >= self.grid_size[0] or next_z < 0 or next_z >= self.grid_size[1]:
            self.last_position = self.position
            self.position = self.position + np.asarray(movement)
            return self.get_state(), -20, True
        self.last_position = self.position
        self.position = self.position + np.asarray(movement)
        self.visited[int(self.position[0]), int(self.position[1])] += 1
        # reward = self.compute_reward()
        reward = self.compute_reward_img_feature()
        done = False
        if (np.min(self.visited) >= self.proper_visit) or (np.max(self.visited) >= 10):
            reward = -10
            done = True
        ## Calculate reward. If out of grid, assign a negative reward
        # if self.droneManagement.if_collision():
            # print('COLLISION')
        return self.get_state(), reward, done

