# coding:utf-8
from cv2 import cv2
import numpy as np
import os
import time

class MYLUT: # my lut filter
    '''
    originlut.shape
    (512, 512, 3) [0, 63]
    '''
    def __init__(self, lutpath='whatthehelllutfile.png'):
        self.loaded = True
        if not os.path.exists(lutpath):
            print('Can not load lut file: {}, try to search in .py path'.format(lutpath))
            lutpath = '{}/{}'.format(os.path.split(os.path.realpath(__file__))[0], lutpath)
            if not os.path.exists(lutpath):
                self.loaded = False
                print('Still can not load lut file: {}'.format(lutpath))
                return
            else:
                print('Using lut file: {}'.format(lutpath))
        lut = cv2.imread(lutpath)
        cube64rows = 8
        cube64size = 64
        cube256size = 256
        cubescale = cube256size / cube64size
        # turn this (64,64,64,3) into (256,256,256,3)
        reshapelut = np.zeros((cube256size, cube256size, cube256size, 3))
        for i in range(cube64size):
            cx = (i % cube64rows) * cube64size
            cy = (i / cube64rows) * cube64size
            cube64 = lut[cy:cy + cube64size, cx:cx + cube64size]
            # some lut is jpg, which need to be fixed
            # cube64[0] = cube64[1]
            # cube64[cube64size - 1] = cube64[cube64size - 2]
            # cube64[:, 0] = cube64[:, 1]
            # cube64[:, cube64size - 1] = cube64[:, cube64size - 2]
            cube62 = cube64[1: cube64size - 1, 1: cube64size - 1]
            cube64 = cv2.resize(cube62, (cube64size, cube64size))

            cube256 = cv2.resize(cube64, (cube256size, cube256size))
            i = i * cubescale
            for k in range(cubescale):
                reshapelut[i + k] = cube256
        self.lut = reshapelut.astype(np.float)

    def imageInLut(self, src, alpha = 1.0):
        if not self.loaded:
            return src
        arr = src.copy()
        arr[:, :] = alpha * self.lut[arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]] + (arr[:, :] * (1 - alpha)) # wow this runs much faster!!
        return arr


def testLut():
    testimg = cv2.imread('/Users/zjj/Desktop/IMG_3271.jpg')
    lutpath = '/Users/zjj/Desktop/lookup_0_origin_2.png'

    lut = MYLUT(lutpath)
    img = testimg.copy()
    img = lut.imageInLut(img)
    cv2.imwrite('/Users/zjj/Desktop/IMG_3271_lut.jpg', img)
    return 0
    # testimg = cv2.imread('/Users/zjj/Downloads/照片/IMG_0495.JPG')\

    # testimg = cv2.resize(testimg, (600, 400))
    fileinlutdir = os.listdir('lut')
    fileinlutdir = sorted(fileinlutdir)
    for filename in fileinlutdir:
        if filename.endswith('.png') and filename.startswith('lookup'):
            lutpath = 'lut/' + filename
            print('Init', time.time())
            lut = MYLUT(lutpath)
            img = testimg.copy()
            for i in range(1):
                print('Start', time.time())
                img = lut.imageInLut(img)
                print('Finish', time.time())
                windowname = filename + str(i)
                cv2.namedWindow(windowname, cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(windowname, 1080, 1920)
                cv2.imshow(windowname, img)
                cv2.waitKey(0)
                cv2.destroyWindow(windowname)
    print("Done")

if __name__ == "__main__":
    testLut()