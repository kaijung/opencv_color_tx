# import the necessary packages
import numpy as np
import cv2


def centroid_histogram(labels_):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(labels_)) + 1)
    (hist, _) = np.histogram(labels_, bins=num_labels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float32")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0

    percent_colors = [(p, c) for (p, c) in zip(hist, centroids)]
    percent_colors = sorted(percent_colors, key=lambda pc: pc[0], reverse=True)

    # loop over the percentage of each cluster and the color of each cluster
    for (percent, color) in percent_colors:
        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50),
                      color.astype("uint8").tolist(), -1)
        start_x = end_x

    # return the bar chart
    return bar


def plot_colors_lab(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0

    percent_colors = [(p, c) for (p, c) in zip(hist, centroids)]
    percent_colors = sorted(percent_colors, key=lambda pc: pc[0], reverse=True)

    # loop over the percentage of each cluster and the color of each cluster
    for (percent, color_lab) in percent_colors:
        color = np.array(color_lab).astype('uint8').reshape(1, 1, 3)
        color = cv2.cvtColor(color, cv2.COLOR_LAB2RGB)

        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50), color[0, 0].tolist(), -1)
        start_x = end_x

    # return the bar chart
    return bar
