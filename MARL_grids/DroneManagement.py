import cv2
import numpy as np
import os
import pprint
import imagehash
import math
import scipy.special as special
import torch
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool
from sensor_msgs.msg import Image


class DroneManagement(object):
    def __init__(self, img_save, drone_name):
        self.pub = rospy.Publisher(drone_name + '/cmd_vel', Twist, queue_size = 1)

        self.takeoff_pub = rospy.Publisher(drone_name + '/takeoff', Empty, queue_size=1)
        self.land_pub = rospy.Publisher(drone_name + '/land', Empty, queue_size=1)
        ## Don't know if this works
        self.navi_home = rospy.Publisher(drone_name + '/autoflight/navigate_home', Bool, queue_size=1)

        rospy.init_node('MARL')

        def __imgCallback__(data):
            img = np.frombuffer(data.data, dtype=np.uint8).reshape((856, 480, -1))
            cv2.imsave(str(self.img_count) + '.jpg', img)

        ## Read image stream
        rospy.Subscriber(drone_name + '/image_raw', Image, __imgCallback__)
        rospy.spin()
        self.speed = rospy.get_param("~speed", 0.5)
        # turn = rospy.get_param("~turn", 1.0)
        self.offset = 1
        ## Suppose the drone never moves on x axis
        self.ind2action = {
            # 0: np.asarray([self.offset, 0, 0]),
            # 1: np.asarray([-self.offset, 0, 0]),
            0: np.asarray([0, self.offset, 0]),
            1: np.asarray([0, -self.offset, 0]),
            2: np.asarray([0, 0, self.offset]),
            3: np.asarray([0, 0, -self.offset])
        }
        ## record the position of the drone. GPS cannot work inside,
        ## thus have to record the position incrementally
        self.position = np.asarray([0, 0, 0])
        self.img_save = img_save
        self.img_count = 0

    def takeoff(self):
        self.pub.publish(self.takeoff_pub)

    def land(self):
        self.pub.publish(self.land_pub)

    def move(self, action):
        vec = self.ind2action[action]
        ## record the position
        self.position += vec
        
        twist = Twist()
        twist.linear.x = vec[0] * self.speed
        twist.linear.y = vec[1] * self.speed
        twist.linear.z = vec[2] * self.speed
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0

        self.pub.publish(twist)
    
    def get_position(self):
        return self.position

    def reset(self):
        self.pub.publish(self.navi_home)

    # def if_collision(self):
    #     return self.client.simGetCollisionInfo().has_collided


class Environment(object):
    def __init__(self, img_save, drone_name, grid_len, grid_size, proper_visit=5):
        self.grid_len = grid_len
        self.grid_size = grid_size
        self.cur_grid = (-1, -1)
        self.droneManagement = DroneManagement(img_save, drone_name)
        self.visited = np.zeros(grid_size)
        ## This defines how many times a grid should be visited
        self.proper_visit = proper_visit

    def reset(self):
        self.droneManagement.reset()
        return 0, 0

    def compute_reward(self):
        distance = np.abs(self.proper_visit - self.visited)
        ## Avoid division by 0
        return np.sum(1 / np.maximum(distance, np.ones(self.visited.shape) * 1e-5))
    
    def __is_done__(self, location):
        done = False
        outof_grid = False
        y, z = location
        if (y < 0) or (y > (self.grid_len * self.grid_size[0])) or (z < 0) or (z > (self.grid_len * self.grid_size[1])):
            done, outof_grid = True, True
        if np.min(self.visited) >= self.proper_visit:
            done, outof_grid = True, False
        return done, outof_grid

    def step(self, action):
        self.droneManagement.move(action)
        position = self.droneManagement.get_position()
        grid_y = position[1] // self.grid_len
        grid_z = position[2] // self.grid_len
        self.visited[grid_y, grid_z] += 1
        done, outof_grid = self.__is_done__((position[1], position[2]))
        ## Calculate reward. If out of grid, assign a negative reward
        if outof_grid:
            reward = -5
        else:
            reward = self.compute_reward()
        # if self.droneManagement.if_collision():
            # print('COLLISION')
        return (grid_y, grid_z), reward, done