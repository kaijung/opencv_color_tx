#!/anaconda/bin/python

import cv2
import skimage.io as io
import argparse


default_im_url = 'http://scikit-image.org/docs/stable/_static/img/logo.png'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_url", type=str, default=default_im_url, help="image url to download")
    args = parser.parse_args()

    print 'downloading image from ', args.image_url

    im = io.imread(args.image_url)
    cv2.imshow('image_from_web', im)

    cv2.waitKey(15000)
    cv2.destroyAllWindows()
