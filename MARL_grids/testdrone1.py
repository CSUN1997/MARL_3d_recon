import cv2
import numpy as np
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import threading
import time
import sys

# class UserVision:
#     def __init__(self, vision):
#         self.index = 0
#         self.vision = vision

#     def save_pictures(self, args):
#         #print("saving picture")
#         img = self.vision.get_latest_valid_picture()

#         if (img is not None):
#             filename = "test_image_%06d.png" % self.index

#             cv2.imwrite(filename, img)
#             self.index +=1

bebop = Bebop()
# connect to the bebop
success = bebop.connect(5)
bebop.safe_takeoff(10)
# bebop.safe_takeoff(10)
# bebop.move_relative(0, 0, -0.5, 0)

bebop.safe_land(10)
bebop.disconnect()
# if (success):

#     # start up the video
#     bebopVision = DroneVision(bebop, is_bebop=True)

#     # userVision = UserVision(bebopVision)
#     # bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
#     success = bebopVision.open_video()
#     time.sleep(2)

#     img = bebopVision.get_latest_valid_picture()
#     cv2.imwrite('test1.jpg', img)
#     time.sleep(3)
#     cv2.imwrite('test2.jpg', bebopVision.get_latest_valid_picture())
#     # width = success.get(cv2.CAP_PROP_FRAME_WIDTH)   #float

#     # height = success.get(cv2.CAP_PROP_FRAME_HEIGHT) #float

#     # cx, cy = width/2, height/2

#     # disconnect nicely so we don't need a reboot
#     bebop.disconnect()
#     sys.exit(0)
# else:
#     print("Error connecting to bebop.  Retry")
