import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2
import time
# import numpy as np
import math

i = 0

def get_img(path):
    global i
    response = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
    img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
    cv2.imwrite(path + '{}.png'.format(i), img_rgb)
    i += 1


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

f1 = client.takeoffAsync()
f1.join()

obj_posi = client.simGetObjectPose('OrangeCube').position 
print(obj_posi)
drone_posi = client.getMultirotorState().kinematics_estimated.position
print(drone_posi)

dx = drone_posi.x_val - obj_posi.x_val
dy = drone_posi.y_val - obj_posi.y_val
actual_radius = math.sqrt((dx*dx) + (dy*dy))
angle_to_center = math.atan2(dy, dx)

camera_heading = (angle_to_center - math.pi) * 180 / math.pi 
# print('======================================================')
# print(camera_heading)

get_img('C:/Users/vsadhu/Desktop/temp_imgs/')


client.moveByVelocityZAsync(0, 0, drone_posi.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading))
time.sleep(3)
get_img('C:/Users/vsadhu/Desktop/temp_imgs/')

f1 = client.moveToPositionAsync(-5 + drone_posi.x_val, 5 + drone_posi.y_val, -10 + drone_posi.z_val, 5)
time.sleep(3)


drone_posi = client.getMultirotorState().kinematics_estimated.position
print(drone_posi)

dx = drone_posi.x_val - obj_posi.x_val
dy = drone_posi.y_val - obj_posi.y_val
actual_radius = math.sqrt((dx*dx) + (dy*dy))
angle_to_center = math.atan2(dy, dx)

camera_heading = (angle_to_center - math.pi) * 180 / math.pi 


client.moveByVelocityZAsync(0, 0, drone_posi.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading))
time.sleep(3)
get_img('C:/Users/vsadhu/Desktop/temp_imgs/')


# print(client.simGetObjectPose('OrangeCube'))

client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
