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
    def __init__(self, drone):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.drone = drone
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
    
    def __del__(self):
        self.client.armDisarm(False)
        self.client.reset()
        self.client.enableApiControl(False)    

    def takeoff(self):
        f = self.client.takeoffAsync(vehicle_name=self.drone)
        f.join()

    def get_state(self):
        return self.client.getMultirotorState(vehicle_name=self.drone)


    def move_to_pnt(self, x, y, z, velocity=5):
        f = self.client.moveToPositionAsync(x, y, z, velocity, vehicle_name=self.drone)
        f.join()

        obj_posi = self.client.simGetObjectPose('OrangeCube').position 
        drone_posi = self.client.getMultirotorState().kinematics_estimated.position

        dx = drone_posi.x_val - obj_posi.x_val
        dy = drone_posi.y_val - obj_posi.y_val
        angle_to_center = math.atan2(dy, dx)

        camera_heading = (angle_to_center - math.pi) * 180 / math.pi 
        f = self.client.moveByVelocityZAsync(
            0, 0, drone_posi.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(False, camera_heading)
        )
        f.join()

    
    def get_img(self):
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 4
        img_rgb = img1d.reshape(response.height, response.width, 3)
        return img_rgb
    
    def get_location(self):
        # position = self.client.getMultirotorState(vehicle_name=self.drone).gps_location
        # return position.latitude, position.longitude, position.altitude
        position = self.client.getMultirotorState().kinematics_estimated.position
        return position.x_val, position.y_val, position.z_val

    def reset(self):
        # self.client.armDisarm(False, self.drone)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def if_collision(self):
        return self.client.simGetCollisionInfo().has_collided


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
        img.dtype = np.uint8
        return self.hash_(Image.fromarray(img))

    def __hex2hash__(self, hex_number):
        ## hex_number could be str(imagehash.ImageHash object)
        return imagehash.hex_to_hash(hex_number)

    def insert(self, img):
        self.imgs.add(str(self.__img2hash__(img)))
        cv2.imwrite('imgs/' + str(len(self.imgs)) + '.png', img)
        if len(self.imgs) % 10 == 0:
            print('--{} imgs saved'.format(len(self.imgs)))

    def difference(self, img):
        imghash = self.__img2hash__(img)
        dist = 0
        for imgstr in self.imgs:
            dist += np.abs(imghash - self.__hex2hash__(imgstr))
        return dist / len(self.imgs)


class Environment(object):
    def __init__(self, imgDB, drone=''):
        print('Initializing env')
        self.droneManagement = DroneManagement(drone)
        self.imgDB = imgDB
        self.offset = 3
        self.actions = {
            0: lambda x, y, z: (x + self.offset, y, z),
            1: lambda x, y, z: (x - self.offset, y, z),
            2: lambda x, y, z: (x, y + self.offset, z),
            3: lambda x, y, z: (x, y - self.offset, z),
            4: lambda x, y, z: (x, y, z + self.offset),
            5: lambda x, y, z: (x, y, z - self.offset)
        }

    def reset(self):
        print('RESETTING')
        self.droneManagement.reset()
        self.droneManagement.takeoff()

    def compute_reward(self, position, next_state, alpha=2, beta=5):
        obj_position = self.droneManagement.client.simGetObjectPose('OrangeCube').position
        obj_position = (obj_position.x_val, obj_position.y_val, obj_position.z_val)

        avg_difference = self.imgDB.difference(next_state) / len(self.imgDB)
        difference_measure = 1 - np.exp(-avg_difference)

        distance = np.sqrt((position[0] - obj_position[0]) ** 2 + (position[1] - obj_position[1]) ** 2\
            + (position[2] - obj_position[2]) ** 2)
        # norm_distance = 1 - np.exp(-distance)
        # distance_measure = (norm_distance ** (alpha - 1) * (1 - norm_distance) ** (beta - 1) / special.beta(alpha, beta)) / (beta / alpha)
        distance_measure = 0
        if distance >= 10:
            distance_measure = -0.5
        elif distance <= 2:
            distance_measure = -0.5
        else:
            distance_measure = 1

        reward = difference_measure + distance_measure
        return torch.Tensor([reward])

    def step(self, action):
        location = self.droneManagement.get_location()
        new_location = self.actions[action](*location)
        self.droneManagement.move_to_pnt(*new_location)
        new_img = self.droneManagement.get_img()
        self.imgDB.insert(new_img)
        ## Calculate reward
        reward = self.compute_reward(self.droneManagement.get_location(), new_img)
        if self.droneManagement.if_collision():
            print('COLLISION')
            return new_img, True, -10

        return new_img, False, reward

    def cur_state(self):
        return self.droneManagement.get_img()

