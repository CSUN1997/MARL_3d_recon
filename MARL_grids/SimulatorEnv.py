import airsim
import cv2
import numpy as np
import os
import pprint
import math
from PIL import Image
import scipy.stats as stats
import torch


class Environment(object):
    def __init__(self, grid_size, gird_len, proper_visit=5, max_visit=10):
        self.grid_len = grid_size
        self.grid_size = grid_size
        self.position = np.zeros(2, dtype=int)
        self.visited = np.zeros(grid_size, dtype=int)
        self.proper_visit = proper_visit
        self.max_visit = max_visit
        self.features = np.zeros(grid_size)

        ## Initialize multi-rotor client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.ind2action = {
            0: [self.grid_len, 0],
            1: [-self.grid_len, 0],
            2: [0, self.grid_len],
            3: [0, -self.grid_len]
        }
        
        self.cur_img = None
        self.sift = cv2.xfeatures2d.SIFT_create()

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

    def compute_reward_img_feature(self):
        y = int(self.position[0])
        z = int(self.position[1])
        visited_times = self.visited[y, z]
        if visited_times == 0:
            return 1
        elif visited_times <= self.proper_visit:
            ## TELL THE AGENT THAT IT IS OK TO STAY IN THE FEATURE-TICH REGION A LITTLE BIT LONGER
            # len_keypnts = self.fake_keypnts[y, z] + int(np.floor(self.fake_keypnts[y, z] * (np.random.random() - 0.3)))
            kps = self.sift.detect(self.cur_img)
            len_kps = len(kps)

            ## Calculate the imcremental mean of the features
            last_mu = self.features[y, z]
            n = self.visited[y, z]
            mu = (len_kps + n * last_mu - last_mu) / n
            # mu = len_keypnts // 500
            self.features[y, z] = mu
            '''
            While designing this reward, to make it generalize (can fit in a new env quickly) and scalable (return rewards from 0 to 1),
            I use a rank score instead of directly calculate a numerical reward.
            '''
            score = stats.percentileofscore(self.features.ravel(), mu) / 100
            return score
        elif visited_times > self.proper_visit:
            return -1

    def get_location(self):
        '''Get the real location of the drone in the simulator instead of the position w.r.t. girds.'''
        # position = self.client.getMultirotorState(vehicle_name=self.drone).gps_location
        # return position.latitude, position.longitude, position.altitude
        position = self.client.getMultirotorState().kinematics_estimated.position
        return position.y_val, position.z_val

    def get_state(self):
        ## 2 means y and z, the coordinates
        state = np.zeros(2 + self.grid_size[0] * self.grid_size[1], dtype=np.float)
        state[0], state[1] = float(self.position[0]), float(self.position[1])
        state[2:] = self.features.ravel()
        return state

    def get_img(self):
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb

    def step(self, action):
        '''DIFFERENCE BETWEEN REAL POSITION AND GRID POSITION'''
        movement = self.ind2action[action]
        next_y = self.position[0] + movement[0]
        next_z = self.position[1] + movement[1]
        
        ## Judge if the drone is out of grid
        if (next_y < 0) or (next_y >= self.grid_size[0]) or (next_z < 0) or (next_z >= self.grid_size[1]):
            return None, -1, True
        ## Take action
        self.position = self.position + np.asarray(movement)
        cur_loc = self.get_location()
        self.client.moveToPositionAsync(0, cur_loc[0] + movement[0] * self.grid_len, cur_loc[1] + movement[1] * self.grid_len, 5)
        self.cur_img = self.get_img()

        reward = self.compute_reward_img_feature()
        done = False
        if np.min(self.visited) >= self.proper_visit:
            done = True
        if np.max(self.visited) >= self.max_visit:
            done = True
            reward = -1

        return self.get_state(), reward, done