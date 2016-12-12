#!/anaconda/bin/python

import cv2
import numpy as np
import utils


MIN_LIGHT_VALUE = 10


def color_kmeans(src_im, n_clusters=3):
    rows, cols, channels = src_im.shape
    img_lab = cv2.cvtColor(src_im, cv2.COLOR_BGR2LAB)
    img_lab = img_lab.reshape((rows * cols, channels))
    # print 'img_lab:', img_lab.shape, img_lab.dtype

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centroids = cv2.kmeans(np.float32(img_lab), n_clusters, None, criteria, 10, flags)

    hist = utils.centroid_histogram(labels)
    hist_colors = [(p, c) for (p, c) in zip(hist, centroids)]
    hist_colors = sorted(hist_colors, key=lambda pc: pc[0], reverse=True)

    stddev = np.sqrt(compactness / (rows * cols))
    # print 'stddev:', stddev, "centers:", percent_colors[1]
    filtered_colors = [pc for pc in hist_colors if pc[1][0] > MIN_LIGHT_VALUE]
    return [colors[0] for colors in filtered_colors], [colors[1] for colors in filtered_colors], stddev


def color_transfer(src_mean, src_stddev, dst_img, dst_masks):
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
        # for debug
        # tmp_im = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
        # cv2.imwrite('images/tmp_im.jpg', tmp_im)

        img_fg = cv2.bitwise_and(out, out, mask=mask)
        out_images.append(img_fg)

    out_img = np.zeros(dst_lab.shape, dtype='uint8')
    for out in out_images:
        out_img = cv2.bitwise_or(out_img, out)
    return cv2.cvtColor(out_img, cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    img_dir = 'images'
    src = cv2.imread('%s/%s' % (img_dir, 'fruits.jpg'))
    dst = cv2.imread('%s/%s' % (img_dir, 'shoe.png'))

    mask_im = cv2.imread('%s/%s' % (img_dir, 'shoe_mask.png'))
    mask1_im = cv2.imread('%s/%s' % (img_dir, 'shoe_mask1.png'))

    hist, centers, stddev = color_kmeans(src, n_clusters=5)
    bar = utils.plot_colors_lab(hist, centers)

    mask_gray = cv2.cvtColor(mask_im, cv2.COLOR_BGR2GRAY)
    mask1_gray = cv2.cvtColor(mask1_im, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    ret, mask1 = cv2.threshold(mask1_gray, 10, 255, cv2.THRESH_BINARY)
    full_mask = cv2.bitwise_or(mask, mask1)
    full_mask_inv = cv2.bitwise_not(full_mask)

    # output = color_transfer2(centers, stddev, dst, mask1)
    output = color_transfer(centers, stddev, dst, (mask, mask1))

    img_fg = cv2.bitwise_and(output, output, mask=full_mask)
    img_bg = cv2.bitwise_and(dst, dst, mask=full_mask_inv)
    output = cv2.add(img_bg, img_fg)

    bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
    output[0:bar.shape[0], 0:bar.shape[1]] = bar

    # cv2.imwrite('%s/%s' % (img_dir, 'output.jpg'), output)
    cv2.imshow('color transfer', output)

    cv2.waitKey(15000)
    cv2.destroyAllWindows()

