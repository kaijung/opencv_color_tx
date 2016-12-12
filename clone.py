#!/anaconda/bin/python

import cv2
import numpy as np


# print "version:", cv2.__version__

# Read images : src image will be cloned into dst
im = cv2.imread("images/ocean_sunset.jpg")
obj = cv2.imread("images/shoe.png")

# Create an all white mask
# mask = 255 * np.ones(obj.shape, obj.dtype)
mask = cv2.imread("images/shoe_full_mask1.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# The location of the center of the src in the dst
width, height, channels = im.shape
center = (height/2, width/2)

# Seamlessly clone src into dst and put the results in output
cloned = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
# cloned = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

# Write results
# cv2.imwrite("images/opencv-normal-clone-example.jpg", normal_clone)
cv2.imwrite("images/mixed-clone.jpg", cloned)

cv2.imshow('mixed clone', cloned)
cv2.waitKey(15000)
cv2.destroyAllWindows()
