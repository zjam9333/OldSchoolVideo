# coding:utf-8
from cv2 import cv2
import numpy as np
import os
import time
import argparse
from tqdm import tqdm
from zzlut import MYLUT 
import videoaudiomix

# define some vars
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type = str, default = '/Users/jam/Desktop/test.mp4')
parser.add_argument("-o", "--output", type = str, default = '/Users/jam/Desktop/test_out.MP4')
parser.add_argument("-lut", "--lutpath", type = str, default = 'lut/lookup_vx.png')
parser.add_argument("-height", "--perferheight", type = int, default = 480)
parser.add_argument("-fps", "--framepersecond", type = int, default = 30)
parser.add_argument("-x264", "--encode264", type = int, default = 1)
userArgs = vars(parser.parse_args())

# lut
lut = MYLUT(lutpath=userArgs['lutpath'])

def USM(src):
    val = 4
    blur = cv2.GaussianBlur(src, (5, 5), 0)
    res = cv2.addWeighted(src, val, blur, 1.0 - val, 0)
    return res

def purpleFringe(src, move = 1):
    copy = src
    cols = copy.shape[1]
    toLeft = copy[:, :, 2]
    toLeft[:, :cols - move] = toLeft[:, move:]
    return copy

def handleImage(src):
    copy = src.copy()
    copy = purpleFringe(copy)
    copy = cv2.GaussianBlur(copy, (5, 5), 0)
    copy = USM(copy)
    copy = lut.imageInLut(copy, 1)
    copy[0::2] = copy[1::2]
    return copy
'''
    # vx ? by code
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
    return copy
'''

def progressVideo(src, output, encodewith264, framepersecond, perferheight):
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

    videocoder = cv2.VideoWriter_fourcc(*'avc1')
    videowriter = cv2.VideoWriter(cacheVideoName, videocoder, fps, storesize)
    if videowriter.isOpened() == False:
        print('Fail to open video writer')
        return False

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
        videowriter.write(frame)
    progressloop.close()
    cap.release()
    videowriter.release()
    print('Finish progress')
    
    # mix video and audio
    videoaudiomix.mixVideoAudio(cacheVideoName, src, output, fps=framepersecond)
    
    print("Cleaning cache")
    os.system("rm -rf {}*".format(cacheNamePrefix))

    print("Finish remix")
    print("Done")

def progressImage(src, output = 'out.jpg'):
    print("Progressing")
    img = cv2.imread(src)
    img = cv2.resize(img, (640, 480))
    vhs = handleImage(img)
    cv2.imshow('output', vhs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done")

if __name__ == "__main__":
    # progressImage('/Users/zjj/Downloads/IMG_A942617EA720-1.jpeg')
    # exit()
    progressVideo(userArgs["input"], output = userArgs["output"], encodewith264 = userArgs["encode264"], perferheight = userArgs["perferheight"], framepersecond = userArgs["framepersecond"])
