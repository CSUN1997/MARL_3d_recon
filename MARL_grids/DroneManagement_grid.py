import numpy as np

class FakeEnv(object):
    def __init__(self, grid_size, proper_visit=5):
        self.grid_size = grid_size
        self.position = np.zeros(2)
        self.visited = np.zeros((grid_size[0] + 2, grid_size[1] + 2), dtype=int)
        ## By grids
        self.ind2action = {
            0: (0, 1),
            1: (0, -1),
            2: (1, 0),
            3: (-1, 0)
        }

        ## This defines how many times a grid should be visited
        self.proper_visit = proper_visit

    def reset(self):
        self.position = np.zeros(2)
        # self.visited = np.zeros((self.grid_size[0] + 2, self.grid_size[1] + 2), dtype=int)
        return self.position

    def compute_reward(self):
        visited_times = self.visited[int(self.position[0]), int(self.position[1])]
        if visited_times == 0:
            return 10
        elif visited_times <= self.proper_visit:
            return 5
        elif visited_times > self.proper_visit:
            return -1
        # distance = np.abs(self.proper_visit - self.visited)
        # ## Avoid division by 0
        # return np.sum(1 / np.maximum(distance, np.ones(self.visited.shape) * 1e-5))
        # if self.position[0] % 2 == 0:
        #     return 10
        # else:
        #     return -10
    
    def __is_done__(self):
        done = False
        outof_grid = False
        y, z = self.position[0], self.position[1]
        if (y < 0) or (y >= self.grid_size[0]) or (z < 0) or (z >= self.grid_size[1]):
            done, outof_grid = True, True
        if np.min(self.visited) >= self.proper_visit:
            done, outof_grid = True, False
        return done, outof_grid

    def step(self, action):
        movement = self.ind2action[action]
        self.position = self.position + np.asarray(movement)
        # position = self.droneManagement.get_position()
        # grid_y = position[1] // self.grid_len
        # grid_z = position[2] // self.grid_len
        # print(self.position)
        self.visited[int(self.position[0]) + 1, int(self.position[1]) + 1] += 1
        done, outof_grid = self.__is_done__()
        ## Calculate reward. If out of grid, assign a negative reward
        if outof_grid:
            reward = -10
        else:
            reward = self.compute_reward()
        # if self.droneManagement.if_collision():
            # print('COLLISION')
        return self.position, reward, done