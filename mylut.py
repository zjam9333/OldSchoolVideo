from cv2 import cv2
import numpy as np
import time

class MYLUT:
    '''
    originlut.shape
    (512, 512, 3) [0, 63]
    '''
    def __init__(self, lutpath='lut/lookup_my.png'):
        lut = cv2.imread(lutpath)
        cube64rows = 8
        cube64size = 64
        # cube256rows = 16
        cube256size = 256
        cubescale = cube256size / cube64size
        reshapelut = np.zeros((cube256size, cube256size, cube256size, 3))
        # largerlut = np.zeros((cube256rows * cube256size, cube256rows * cube256size, 3), dtype=np.uint8)
        for i in range(cube64size):
            cx = (i % cube64rows) * cube64size
            cy = (i / cube64rows) * cube64size
            cube64 = lut[cy:cy + cube64size, cx:cx + cube64size]
            _rows, _cols, _ = cube64.shape
            if _rows == 0 or _cols == 0:
                continue
            cube256 = cv2.resize(cube64, (cube256size, cube256size))
            i = i * cubescale
            for k in range(cubescale):
                reshapelut[i + k] = cube256
                # create a lager lut image
                # index = i + k
                # index_y = (index / cube256rows) * cube256size
                # index_x = (index % cube256rows) * cube256size
                # largerlut[index_y: index_y+cube256size, index_x: index_x + cube256size] = cube256
        self.lut = reshapelut
        # cv2.imshow('largetlut', cv2.resize(largerlut, (512, 512)))
        # cv2.waitKey(0)

    def imageInLut(self, src):
        arr = src.copy()
        bs = arr[:, :, 0]
        gs = arr[:, :, 1]
        rs = arr[:, :, 2]
        arr[:, :] = self.lut[bs, gs, rs] # wow this runs much faster!!
        return arr

        # img = src.reshape(-1, 3)
        # for iy in range(img.shape[0]):
        #     b,g,r = img[iy]
        #     img[iy] = self.lut[b, g, r]
        # return img.reshape(src.shape)

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    print('init', time.time())
    lut = MYLUT()
    print('start', time.time())
    img = lut.imageInLut(img)
    print('finish', time.time())
    # cv2.imwrite(output, vhs)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")