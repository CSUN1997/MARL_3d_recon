import cv2
import numpy as np
import matplotlib.pyplot as plt

def SIFT(img, is_gray=False):
    gray = img if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    kp = sift.detect(gray, None)
#     img = cv2.drawKeypoints(gray, kp, img)

    ## Generate keypnt features
    kp, des = sift.compute(gray, kp)
    
#     plt.imsave('./sift.png', img)
    return kp, des


def feature_match(img1, img2, kp1, kp2, des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des_l, des_r)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x: x.distance)
#     # Draw first 10 matches.
#     img3 = cv2.drawMatches(left, kp_l, right, kp_r,
#                            matches[:10],None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     plt.imsave('./match.png', img3)
    return matches


if __name__ == '__main__':
    left = cv2.imread('./imgs/left.jpg')
    right = cv2.imread('./imgs/right.jpg')
    kp_l, des_l = SIFT(left)
    kp_r, des_r = SIFT(right)
    ## Don't forget the dtype casting
    des_l.dtype = np.uint8
    des_r.dtype = np.uint8

    matches = feature_match(left, right, kp_l, kp_r, des_l, des_r)