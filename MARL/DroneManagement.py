import airsim
import cv2
import numpy as np
import os
import pprint

class DroneManagement:
    def __init__(self, drone_names):
        '''
        drone_names should be identical to the settings.json file
        '''
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.drone_names = drone_names
        for drone in self.drone_names:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
    
    def __del__(self):
        for drone in self.drone_names:
            self.client.enableApiControl(False, drone)    

    def __takeoff__(self, drone_name):
        f = self.client.takeoffAsync(vehicle_name=drone_name)
        f.join()

    def all_take_off(self):
        for drone in self.drone_names:
            self.__takeoff__(drone)

    def get_state(self, drone_name):
        return self.client.getMultirotorState(vehicle_name=drone_name)

    def move_to_pnt(self, x, y, z, velocity, drone_name):
        f = self.client.moveToPositionAsync(x, y, z, velocity, vehicle_name=drone_name)
        f.join()
    
    def get_img(self, drone_name):
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)], vehicle_name=drone_name)
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb

    def reset(self):
        for drone in self.drone_names:
            self.client.armDisarm(False, drone)
        self.client.reset()
