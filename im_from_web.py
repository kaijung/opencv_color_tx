#!/anaconda/bin/python

import cv2
import numpy as np
import skimage.io as io
from PIL import Image
import requests
import argparse
# import time


DOWNLOAD_TIMEOUT = 3


default_im_url = 'http://scikit-image.org/docs/stable/_static/img/logo.png'


def skim_dl(img_url):
    return io.imread(img_url)


def requests_dl(img_url):
    img = np.array(Image.open(requests.get(img_url, stream=True, timeout=DOWNLOAD_TIMEOUT).raw))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_url", type=str, default=default_im_url, help="image url to download")
    args = parser.parse_args()

    print 'downloading image from ', args.image_url

    # im = skim_dl(args.image_url)
    im = requests_dl(args.image_url)
    cv2.imshow('image_from_web', im)

    cv2.waitKey(15000)
    cv2.destroyAllWindows()
