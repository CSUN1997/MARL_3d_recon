import numpy as np

class FakeEnv(object):
    def __init__(self, grid_size, proper_visit=5):
        self.grid_size = grid_size
        self.position = np.zeros(2)
        self.visited = np.zeros((grid_size[0], grid_size[1]), dtype=int)
        ## By grids
        self.ind2action = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }

        self.last_position = self.position

        ## This defines how many times a grid should be visited
        self.proper_visit = proper_visit

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

    # def compute_reward_feature(self, img):

    

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
        # position = self.droneManagement.get_position()
        # grid_y = position[1] // self.grid_len
        # grid_z = position[2] // self.grid_len
        # print(self.position)
        self.visited[int(self.position[0]), int(self.position[1])] += 1
        reward = self.compute_reward()
        done = False
        if (np.min(self.visited) >= self.proper_visit) or (np.max(self.visited) >= 10):
            reward = -10
            done = True
        ## Calculate reward. If out of grid, assign a negative reward
        return self.get_state(), reward, done