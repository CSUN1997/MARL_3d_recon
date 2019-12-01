import airsim
import cv2
import numpy as np
import os
import pprint
import imagehash
import math
from PIL import Image
import scipy.special as special
import torch

class DroneManagement(object):
    def __init__(self, client, drone_name):
        self.client = client
        self.drone_name = drone_name

    def takeoff(self):
        f = self.client.takeoffAsync(vehicle_name=self.drone_name)
        f.join()

    def move_to_pnt(self, x, y, z, velocity=5):
        f = self.client.moveToPositionAsync(x, y, z, velocity, vehicle_name=self.drone_name)
        f.join()
    
    def get_img(self):
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],\
            vehicle_name=self.drone_name)[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb
    
    def get_location(self):
        # position = self.client.getMultirotorState(vehicle_name=self.drone).gps_location
        # return position.latitude, position.longitude, position.altitude
        position = self.client.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated.position
        return position.y_val, position.z_val 

    # def reset(self):
    #     # self.client.armDisarm(False, self.drone)
    #     self.client.reset()
    #     self.client.enableApiControl(True)
    #     self.client.armDisarm(True)

    def if_collision(self):
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or (collision_info.object_id != -1):
            # print(collision_info)
            print(self.client.getMultirotorState().landed_state)
            return True
        return False 


class Environment(object):
    def __init__(self, ):
        print('Initializing env')
        self.droneManagement = DroneManagement()
        self.offset = 3
        self.position = np.zeros(2)
        self.last_postion = self.position
        self.cur_img = None
        self.ind2action = {
            0: [self.offset, 0],
            1: [-self.offset, 0],
            2: [0, self.offset],
            3: [0, -self.offset]
        }

    def reset(self):
        print('RESETTING')
        self.droneManagement.reset()
        self.droneManagement.takeoff()

    def compute_reward(self):
        

    def get_state(self):
        img = self.droneManagement.get_img()
        self.cur_img = img
        return (img, self.droneManagement.get_location())

    def step(self, action):
        ## Take action
        self.last_postion = self.position
        self.position = self.position + np.asarray(self.ind2action[action])
        self.droneManagement.move_to_pnt(0, self.position[0], self.position[1])

        reward = self.compute_reward()
        if self.droneManagement.if_collision():
            print('COLLISION')
            return new_img, True, torch.Tensor([-10.])
        elif len(self.imgDB) >= 100:
            ## When the number of images exceeds 50
            return new_img, True, reward

        return new_img, False, reward

    def cur_state(self):
        return self.droneManagement.get_img()

