#!/anaconda/bin/python

import cv2
import numpy as np
import argparse
import utils


MIN_LIGHT_VALUE = 15
MAX_LIGHT_VALUE = 240
MAX_SAMPLE_COLS = 300


def color_kmeans(src_bg_im, n_colors=3):
    """
    compute k-means color clustering
    :param src_bg_im:
    :param n_colors: number of object's color
    :return: histogram, colors and colors' stddev
    """
    src_bg_im = resize_if_necessary(src_bg_im, MAX_SAMPLE_COLS)

    rows, cols, channels = src_bg_im.shape
    img_lab = cv2.cvtColor(src_bg_im, cv2.COLOR_BGR2LAB)
    img_lab = img_lab.reshape((rows * cols, channels))

    # print 'src_im:', rows, cols, src_im.dtype
    # print 'img_lab:', img_lab.shape, img_lab.dtype

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    clusters = n_colors + 2
    compactness, labels, centroids = cv2.kmeans(np.float32(img_lab), clusters, None, criteria, 10, flags)

    hist = utils.centroid_histogram(labels)
    hist_colors = [(p, c) for (p, c) in zip(hist, centroids)]
    hist_colors = sorted(hist_colors, key=lambda pc: pc[0], reverse=True)

    stddev = np.sqrt(compactness / (rows * cols))
    filtered_colors = [pc for pc in hist_colors if pc[1][0] > MIN_LIGHT_VALUE and pc[1][0] < MAX_LIGHT_VALUE]
    if len(filtered_colors) < n_colors:
        filtered_colors = hist_colors

    return [colors[0] for colors in filtered_colors], [colors[1] for colors in filtered_colors], stddev


def color_transfer(src_mean, src_stddev, dst_img, dst_masks):
    """
    transfer colors of destination image based on colors of the source
    :param src_mean:
    :param src_stddev:
    :param dst_img:
    :param dst_masks:
    :return: color-transferred image
    """
    dst_lab = cv2.cvtColor(dst_img, cv2.COLOR_BGR2LAB)
    _, _, channels = dst_lab.shape

    src_stddev = np.array([src_stddev])

    # Find mean and std of each channel for each image src_mean, dst_mean, src_stddev, dst_stddev
    out_images = []
    for i, mask in  enumerate(dst_masks):
        dst_mean, dst_stddev = cv2.meanStdDev(dst_lab, mask=mask)
        # Split into individual channels
        dst_channels = cv2.split(dst_lab)

        # For each channel calculate the color distribution
        for ch in xrange(channels):
            dst_channels[ch] = dst_channels[ch] - dst_mean[ch]
            dst_channels[ch] = dst_channels[ch] * (dst_stddev[ch] / src_stddev)
            dst_channels[ch] = np.clip(dst_channels[ch] + src_mean[i][ch], 0, 255)

        out = np.uint8(cv2.merge(dst_channels))
        im_fg = cv2.bitwise_and(out, out, mask=mask)
        out_images.append(im_fg)

    out_img = np.zeros(dst_lab.shape, dtype='uint8')
    for out in out_images:
        out_img = cv2.bitwise_or(out_img, out)
    return cv2.cvtColor(out_img, cv2.COLOR_LAB2BGR)


def normal_clone(img_fg, img_bg, mask_fg=None):
    """
    merge foreground image with background using normal cloning algorithm
    :param img_fg:
    :param img_bg:
    :param mask_fg:
    :return: merged image
    """
    rows_fg, cols_fg, _ = img_fg.shape
    rows_bg, cols_bg, _ = img_bg.shape

    # Create an all white mask
    if mask_fg is None:
        mask_fg = 255 * np.ones(img_fg.shape, img_fg.dtype)

    # print 'fg shape:', rows_fg, cols_fg
    # print 'bg shape:', rows_bg, cols_bg
    # print 'half_cols_bg:', half_cols_bg

    half_cols_bg = int(cols_bg / 2)
    if cols_fg > half_cols_bg:
        ratio = float(half_cols_bg) / cols_fg
        size = (half_cols_bg, int(ratio * rows_fg))
        img_fg = cv2.resize(img_fg, size)
        mask_fg = cv2.resize(mask_fg, size)

    # The location of the center of the src in the dst
    center = (int(cols_bg / 2), int(rows_bg / 2))
    # center = (int(0.5 * cols_bg), int(0.4 * rows_bg))

    # Seamlessly clone src into dst and put the results in output
    cloned_im = cv2.seamlessClone(img_fg, img_bg, mask_fg, center, cv2.NORMAL_CLONE)
    return cloned_im


def mask_clone(img_fg, img_bg, mask_fg):
    """
    merge foreground image with background
    :param img_fg:
    :param img_bg:
    :param mask_fg:
    :return: merged image
    """
    rows_fg, cols_fg, _ = img_fg.shape
    rows_bg, cols_bg, _ = img_bg.shape

    half_cols_bg = cols_bg / 2
    if cols_fg > half_cols_bg:
        ratio = float(half_cols_bg) / cols_fg
        size = (half_cols_bg, int(ratio * rows_fg))
        img_fg = cv2.resize(img_fg, size)
        img_bg = cv2.resize(img_bg, size)
        mask_fg = cv2.resize(mask_fg, size)

    mask_inv = cv2.bitwise_not(mask_fg)

    img_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask_fg)
    img_bg = cv2.bitwise_and(img_bg, img_bg, mask=mask_inv)
    mask_img = cv2.add(img_bg, img_fg)
    return mask_img


def resize_if_necessary(src_im, max_size=800):
    """
    resize image if image's columns > max_cols
    :param src_im:
    :param max_size:
    :return: resized image if need
    """
    rows, cols, channels = src_im.shape
    if cols > max_size or rows > max_size:
        if cols > rows:
            ratio = float(max_size) / cols
            size = (max_size, int(ratio * rows))
        else:
            ratio = float(max_size) / rows
            size = (int(ratio * cols), max_size)
        src_im = cv2.resize(src_im, size)

    return src_im


def print_time(msg, start_t):
    """
    print performance info
    :param msg:
    :param start_t:
    :return:
    """
    end_t = cv2.getTickCount()
    took_time = 1000 * (end_t - start_t) / cv2.getTickFrequency()
    print '%s. I took %d ms' % (msg, took_time)


def process_images(src, dst, color_masks_im, full_mask_im):
    src = resize_if_necessary(src, max_size=1024)

    color_masks = [cv2.cvtColor(msk_im, cv2.COLOR_BGR2GRAY) for msk_im in color_masks_im]
    # use number of colors as clusters
    n_clusters = len(color_masks)

    start_t = cv2.getTickCount()
    hist, centers, stddev = color_kmeans(src, n_colors=n_clusters)

    print_time('Color Kmeans done', start_t)

    full_mask = cv2.cvtColor(full_mask_im, cv2.COLOR_BGR2GRAY)

    merged_mask = reduce(cv2.bitwise_or, color_masks)
    merged_mask_inv = cv2.bitwise_not(merged_mask)

    output = color_transfer(centers, stddev, dst, color_masks)

    print_time('Color transferred', start_t)

    img_fg = cv2.bitwise_and(output, output, mask=merged_mask)
    img_bg = cv2.bitwise_and(dst, dst, mask=merged_mask_inv)
    output = cv2.add(img_bg, img_fg)

    print_time('Image merged', start_t)

    cloned = normal_clone(output, src, full_mask)
    # cloned = mask_clone(output, src, full_mask)
    print_time('Image cloned', start_t)
    bar = utils.plot_colors_lab(hist, centers)
    bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

    return cloned, output, bar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bg_image", type=str, default="gjl.png", help="background image")
    args = parser.parse_args()

    img_dir = 'images'
    src = cv2.imread('%s/%s' % (img_dir, args.bg_image))
    dst = cv2.imread('%s/%s' % (img_dir, 'shoe.png'))

    mask1_im = cv2.imread('%s/%s' % (img_dir, 'shoe_mask1.png'))
    mask_im = cv2.imread('%s/%s' % (img_dir, 'shoe_mask.png'))
    full_mask_im = cv2.imread('%s/%s' % (img_dir, 'shoe_full_mask1.png'))

    cloned, output, bar = process_images(src, dst, (mask1_im, mask_im), full_mask_im)

    cv2.imshow('cloned', cloned)
    cv2.imwrite('%s/%s' % (img_dir, 'output_shoe.png'), cloned)

    output[0:bar.shape[0], 0:bar.shape[1]] = bar
    cv2.imshow('color transfer', output)
    cv2.imwrite('%s/%s' % (img_dir, 'colored_shoe.png'), output)

    cv2.waitKey(15000)
    cv2.destroyAllWindows()
