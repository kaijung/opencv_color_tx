#!/usr/bin/env python

import cv2
import numpy as np
from PIL import Image

img = cv2.imread('messi5.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

hue = hsv_img.T[0]
sat = hsv_img.T[1]
val = hsv_img.T[2]

sl = np.repeat(np.uint8(120), hue.shape[0] * hue.shape[1])
sl = sl.reshape(hue.shape)

img = np.array((hue, sat, val)).T
# img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

Image.fromarray(img).save('output_messi5.jpg')
