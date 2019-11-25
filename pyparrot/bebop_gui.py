"""
Demo of the Bebop vision using DroneVisionGUI (relies on libVLC).  It is a different
multi-threaded approach than DroneVision

Author: Amy McGovern
"""
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI
import threading
import cv2
import time
from PyQt5.QtGui import QImage
import sys, getopt

import flying_routines
import vision_routines

isAlive = False
isVision = False

if __name__ == "__main__":

    try:
        args = sys.argv
        if(len(args) == 1): raise Exception('Not enough input arguments')

        argumentlist = args[1:]
        unixoptions="f:v:h"
        gnuoptions=["fly=","vision=","help"]
        arguments, vals = getopt.getopt(argumentlist, unixoptions, gnuoptions)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    for arg, val in arguments:
        if arg in ('-f', '--fly'):
            fly_func = val
        elif arg in ('-v', '--vision'):
            vision_func = val
            isVision = True
        elif arg in ('-h','--help'):
            print("""This script shows the live-stream video from drone and takes in two inputs.\n \
-f, --fly:    This option is the name of script from flying_routines and is used to control drone\'s flight pattern. \n \
-v, --vision: This option is the name of script from vision_routines and is used to control processing of images from drone.""")
            sys.exit(2)

    # make my bebop object
    bebop = Bebop()

    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=getattr(flying_routines, fly_func),
                                     user_args=(bebop, ))
        # user_draw_window_fn=draw_current_photo)

        if(isVision):
            UserVision = getattr(vision_routines, vision_func)
            userVision = UserVision(bebopVision)
            bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
        bebopVision.open_video()

    else:
        print("Error connecting to bebop.  Retry")
