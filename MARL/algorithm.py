import numpy as np
import imagehash
import cv2
import matplotlib.pyplot as plt
import os

class ImgDB:
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.imghash = []

    def insert(self, img):
        self.imghash.append(imagehash.)