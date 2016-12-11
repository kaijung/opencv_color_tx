#!/anaconda/bin/python

import cv2
import numpy as np


def color_transfer(src_img, dst_img):
    src_lab = np.float32(cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB))
    dst_lab = np.float32(cv2.cvtColor(dst_img, cv2.COLOR_BGR2LAB))
    # src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
    # dst_lab = cv2.cvtColor(dst_img, cv2.COLOR_BGR2LAB)

    # Find mean and std of each channel for each image src_mean, dst_mean, src_stddev, dst_stddev
    src_mean, src_stddev = cv2.meanStdDev(src_lab)
    dst_mean, dst_stddev = cv2.meanStdDev(dst_lab)

    # Split into individual channels
    # src_channels = cv2.split(src_lab)
    dst_channels = cv2.split(dst_lab)

    print 'src_stddev[0]:', type(src_stddev), src_stddev[0]
    print 'src_mean[0]:', type(src_mean[0]), src_mean[0]

    # For each channel calculate the color distribution
    for ch in xrange(3):
        dst_channels[ch] = dst_channels[ch] - dst_mean[ch]
        dst_channels[ch] = dst_channels[ch] * (dst_stddev[ch] / src_stddev[0])
        dst_channels[ch] = np.clip(dst_channels[ch] + src_mean[ch], 0, 255)

    # TODO: mat::convertTo error here
    out_img = np.uint8(cv2.merge(dst_channels))
    return cv2.cvtColor(out_img, cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    # print "version:", cv2.__version__

    img_dir = 'images'
    src = cv2.imread('%s/%s' % (img_dir, 'ocean_day.jpg'))
    dst = cv2.imread('%s/%s' % (img_dir, 'ocean_sunset.jpg'))
    output = color_transfer(src, dst)

    cv2.imwrite('%s/%s' % (img_dir, 'output.jpg'), output)
    cv2.imshow('color transfer', output)
    cv2.waitKey(15000)
    cv2.destroyAllWindows()
