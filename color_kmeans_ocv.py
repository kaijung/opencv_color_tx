#!/anaconda/bin/python

# USAGE
# python color_kmeans_ocv.py --image images/jp.png --clusters 3

# import the necessary packages
# from sklearn.cluster import KMeans
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import utils


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--clusters", required=True, type=int, help="# of clusters")
args = ap.parse_args()

# load the image and convert it from BGR to RGB so that
# we can display it with matplotlib
image = cv2.imread(args.image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

# print 'image.ndim:{}, img_lab.ndim:{}'.format(image.ndim, img_lab.ndim)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS
n_clusters = args.clusters

# Apply KMeans
start_time = time.time()
compactness, labels, centers = cv2.kmeans(np.float32(image), n_clusters, None, criteria, 10, flags)
print 'took time %d ms' % int((time.time()-start_time) * 1000)

hist = utils.centroid_histogram(labels)
bar = utils.plot_colors(hist, centers)

# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
