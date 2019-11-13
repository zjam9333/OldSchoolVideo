# pyyy

from cv2 import cv2
import numpy as np
import os
import time
import argparse
from tqdm import tqdm

defaultlutpath = 'lut/lookup_vx.png'

class MYLUT: # my lut filter
    '''
    originlut.shape
    (512, 512, 3) [0, 63]
    '''
    def __init__(self, lutpath=defaultlutpath):
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
            cube256 = cv2.resize(cube64, (cube256size, cube256size))
            i = i * cubescale
            for k in range(cubescale):
                reshapelut[i + k] = cube256
        self.lut = reshapelut

    def imageInLut(self, src):
        if not self.loaded:
            return src
        arr = src.copy()
        arr[:, :] = self.lut[arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]] # wow this runs much faster!!
        return arr

# define some vars
# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type = str, default = '/Users/zjj/Downloads/2160p.MOV')
parser.add_argument("-o", "--output", type = str, default = '/Users/zjj/Downloads/test.out2.mov')
parser.add_argument("-lut", "--lutpath", type = str, default = defaultlutpath)
parser.add_argument("-height", "--perferheight", type = int, default = 240)
parser.add_argument("-fps", "--framepersecond", type = int, default = 30)
parser.add_argument("-x264", "--encode264", type = int, default = 1)
parser.add_argument("-interlaced", "--interlaced", type = int, default = 0)
userArgs = vars(parser.parse_args())
# lut
lut = MYLUT(lutpath=userArgs['lutpath'])
# sharpen
sharpkernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# purple
purplesize = 1
# blur
blursize = 5

def USM(src):
    val = 2
    blur = cv2.GaussianBlur(src, (blursize, blursize), 3)
    res = cv2.addWeighted(src, val, blur, 1.0 - val, 0)
    return res

def handleImage(src):
    copy = src.copy()
    '''
    # purple edge
    channel = copy[:, :, 1]
    channel[:-purplesize, :-purplesize] = channel[purplesize:, purplesize:]

    # blur
    copy = cv2.GaussianBlur(copy, (blursize, blursize), 0)

    # sharpen 
    copy = cv2.filter2D(copy, -1, sharpkernel)

    # colors 
    # lut
    copy = lut.imageInLut(copy)
    '''
    # copy = cv2.medianBlur(copy, 3)
    # copy = cv2.bilateralFilter(copy, 3, 11, 11)
    # copy = cv2.filter2D(copy, -1, sharpkernel)
    copy = USM(copy)
    copy = lut.imageInLut(copy)
    return copy
'''
    # vx ?
    # make more yellow
    B = copy[:, :, 0]
    B[:, :] = B[:, :] * 0.9
    # change Saturation
    HSV = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    S = HSV[:, :, 1]
    S[:, :] = S[:, :] * 0.6
    # make brighter
    Vfloat = HSV[:, :, 2].astype(np.float)
    Vfloat *= 1.3
    Vfloat += 20
    Vfloat[Vfloat > 255] = 255
    HSV[:, :, 2] = Vfloat.reshape(S.shape)
    copy = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
'''
    # return copy

def progressVideo(src, output, encodewith264, framepersecond, interlaced, perferheight):
    if not os.path.exists(src):
        print('Input video path {} does not exist'.format(src))
        return False
    cap = cv2.VideoCapture(src)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # I prefer 4:3 video
    videoheight = int(perferheight)
    videowidth = int(float(videoheight) / float(height) * float(width))
    if videoheight > videowidth:
        videowidth = videoheight
        videoheight = int(float(videowidth) / float(width) * float(height))
    videosize = (videowidth, videoheight)
    storesize = [videowidth, videoheight]
    rate = float(width) / float(height)
    shouldCropCols = 0
    shouldCropRows = 0
    if rate > (4.0 / 3.0):
        storesize[0] = int(float(videoheight) / 3.0 * 4.0)
        shouldCropCols = int((videowidth - storesize[0]) / 2)
    elif rate < (3.0 / 4.0):
        storesize[1] = int(float(videowidth) / 4.0 * 3.0)
        shouldCropRows = int((videoheight - storesize[1]) / 2)
    storesize = (storesize[0], storesize[1])

    cacheNamePrefix = "cache_will_be_deleted_jowfajoasdfklajdsf"
    # clean old cache if need
    os.system("rm -rf {}*".format(cacheNamePrefix))
    cacheNameSuffix = time.strftime('%Y%m%d_%H%M%S')
    cacheVideoName = "{}_{}.mp4".format(cacheNamePrefix, cacheNameSuffix)

    videocoder = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(cacheVideoName, videocoder, fps, storesize)
    if videowriter.isOpened() == False:
        print('Fail to open video writer')
        return False

    # currentframeindex = 0
    lastframe = np.array([])
    odd = True
    progressloop = tqdm(range(framecount))
    for _ in progressloop:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, videosize)
        # crop frame into storesize
        if shouldCropCols > 0:
            frame = frame[:, shouldCropCols: shouldCropCols + storesize[0]]
        if shouldCropRows > 0:
            frame = frame[shouldCropRows: shouldCropRows + storesize[1]]
        frame = handleImage(frame)
        if interlaced: # fake interlace effect
            if lastframe.any():
                copyframe = frame.copy()
                startRow = 0 if odd else 1
                odd = not odd
                copyframe[startRow::2] = lastframe[startRow::2]
                videowriter.write(copyframe)
            else:
                copyframe = frame.copy()
                copyframe[::2] = copyframe[1::2]
                videowriter.write(copyframe)
            lastframe = frame
        else:
            videowriter.write(frame)
    progressloop.close()
    cap.release()
    videowriter.release()
    print('Finish progress')

    print('Remix audio and video')
    
    # ffmpeg -i INPUT_Video -i INPUT_Audio -map 0:v -map 1:a -c:v copy -c:a copy output.mp4
    ffmpegcommand = "ffmpeg -i {} -i {} -map 0:v -map 1:a -c:v {} -c:a copy {} {}".format(cacheVideoName, src, 'libx264' if encodewith264 else 'copy', '-r {}'.format(framepersecond) if (framepersecond > 0 and encodewith264) else '', output)
    print('FFMpeg command: {}'.format(ffmpegcommand))
    os.system(ffmpegcommand)
    print("Cleaning cache")
    os.system("rm -rf {}*".format(cacheNamePrefix))

    print("Finish remix")
    print("Done")

def progressImage(src, output = 'out.jpg'):
    print("Progressing")
    img = cv2.imread(src)
    # img = cv2.resize(img, (320, 240))#, interpolation=cv2.INTER_AREA)
    vhs = handleImage(img)
    cv2.imshow('output', vhs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done")

def testLut():
    testimg = cv2.imread('/Users/zjj/Downloads/照片/IMG_0495.JPG')
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
    # testLut()
    # exit()
    # progressImage('lut/lookup_0_origin.png')#('/Users/zjj/Desktop/output_screenshot_13.11.2019.png')
    # exit()
    progressVideo(userArgs["input"], output = userArgs["output"], encodewith264 = userArgs["encode264"], perferheight = userArgs["perferheight"], framepersecond = userArgs["framepersecond"], interlaced = userArgs["interlaced"])