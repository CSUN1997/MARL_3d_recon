import airsim
import cv2
import numpy as np
import os
import pprint
import imagehash

class DroneManagement(object):
    def __init__(self, drone):
        '''
        drone_names should be identical to the settings.json file
        '''
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.drone = drone
        self.client.enableApiControl(True, self.drone)
        self.client.armDisarm(True, self.drone)
    
    def __del__(self):
        self.client.enableApiControl(False, self.drone)    

    def takeoff(self):
        f = self.client.takeoffAsync(vehicle_name=self.drone)
        f.join()

    def get_state(self):
        return self.client.getMultirotorState(vehicle_name=self.drone)


    def move_to_pnt(self, x, y, z, velocity=5):
        f = self.client.moveToPositionAsync(x, y, z, velocity, vehicle_name=self.drone)
        f.join()
    
    def get_img(self):
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene)], vehicle_name=self.drone)
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb
    
    def get_location(self):
        ## need to change
        return self.client.getMultirotorState(vehicle_name=self.drone)

    def reset(self):
        self.client.armDisarm(False, self.drone)
        self.client.reset()


class Environment(object):
    def __init__(self, drone):
        self.droneManagement = DroneManagement(drone)
        self.offset = 1
        self.actions = {
            0: lambda x, y, z: (x + self.offset, y, z),
            1: lambda x, y, z: (x - self.offset, y, z),
            2: lambda x, y, z: (x, y + self.offset, z),
            3: lambda x, y, z: (x, y - self.offset, z),
            4: lambda x, y, z: (x, y, z + self.offset),
            5: lambda x, y, z: (x, y, z - self.offset)
        }

    def reset(self):
        self.droneManagement.reset()
        self.droneManagement.takeoff()

    def step(self, action):
        location = self.droneManagement.get_location()
        new_location = self.actions[action](location)
        self.droneManagement.move_to_pnt(*new_location)
        new_img = self.droneManagement.get_img()
        return new_img, False

    def cur_state(self):
        return self.droneManagement.get_img()


class ImgDatabase(object):
    def __init__(self, method):
        str2method = {
            'average': imagehash.average_hash,
            'perception': imagehash.phash,
            'difference': imagehash.dhash
        }
        self.hash_ = str2method[method]

        self.imgs = set()

    def __len__(self):
        return len(self.imgs)

    def __img2hash__(self, img, method='average_hash'):
        return self.hash_(img)

    def __hex2hash__(self, hex_number):
        ## hex_number could be str(imagehash.ImageHash object)
        return imagehash.hex_to_hash(hex_number)

    def insert(self, img):
        self.imgs.add(str(self.__img2hash__(img)))

    def difference(self, img):
        imghash = self.__img2hash__(img)
        dist = 0
        for imgstr in self.imgs:
            dist += np.abs(imghash - self.__hex2hash__(imgstr))
        return dist / len(self.imgs)