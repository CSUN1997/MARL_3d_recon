#this file is intended to store vision routines for bebop drone
class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        pass
        #print("saving picture")
        #img = self.vision.get_latest_valid_picture()

        # limiting the pictures to the first 10 just to limit the demo from writing out a ton of files
        #if (img is not None and self.index <= 10):
        #    filename = "test_image_%06d.png" % self.index
        #    cv2.imwrite(filename, img)
        #    self.index +=1
