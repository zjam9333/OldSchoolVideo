# pyyy

from cv2 import cv2
import numpy as np
import os
import time
import argparse

def VHSImage(src):
    copy = src.copy()
    # blur
    copy = cv2.GaussianBlur(copy, (3, 3), 0)
    # sharpen 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    copy = cv2.filter2D(copy, -1, kernel)

    # change Saturation
    HSV = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    S = HSV[:, :, 1]
    S[:, :] = S[:, :] * 0.8
    copy = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    # translate color's channel
    cols = copy.shape[1]
    move = 1
    # pick one channel to translate, 0:B 1:G 2:R
    toLeft = copy[:, :, 0]
    toLeft[:, :cols - move] = toLeft[:, move:]
    toRight = copy[:, :, 2]
    toRight[:, move:] = toRight[:, :cols - move]
    rows = copy.shape[0]
    toUp = copy[:, :, 1]
    toUp[:rows - move:] = toUp[move:, :]

    # make more yellow
    B = copy[:, :, 0]
    B[:, :] = B[:, :] * 0.9
    G = copy[:, :, 1]
    G[:, :] = G[:, :] * 0.95

    black = np.zeros_like(copy)
    copy = cv2.bitwise_or(copy, black)

    return copy

def doWithVideo(src, output = 'out.mp4'):
    cap = cv2.VideoCapture(src)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total frame: {}'.format(framecount))

    videoheight = int(720)
    videowidth = int(float(videoheight) / float(height) * float(width))
    videosize = (videowidth, videoheight)

    cachePrefixName = time.strftime('%Y%m%d_%H%M%S')
    cacheVideoName = "{}.cachevideo.mp4".format(cachePrefixName)
    cacheAudioName = "{}.cacheaudio.mp3".format(cachePrefixName)

    cc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cacheVideoName, cc, fps, videosize)
    if out.isOpened() == False:
        print('fail to open video writer')
        return False

    currentframeindex = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, videosize)
        frame = VHSImage(frame)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        out.write(frame)
        print('progress:{}/{}'.format(currentframeindex, framecount))
        currentframeindex += 1
    cap.release()
    out.release()
    print('finish progress')

    # ffmpeg -i INPUT_VIDEO -f mp3 -vn OUTPUT_AUDIO
    # ffmpeg -i INPUT_VIDEO -i INPUT_AUDIO -c:v copy OUTPUT_VIDEO
    print('remix audio and video')
    os.system("ffmpeg -i {} -f mp3 -vn {}".format(src, cacheAudioName))
    # check audio file if exists
    if os.path.exists(cacheAudioName):
        # remix audio and video
        os.system("ffmpeg -i {} -i {} -c:v copy {}".format(cacheVideoName, cacheAudioName, output))
        print("cleaning cache")
        os.system("rm -rf {} {}".format(cacheAudioName, cacheVideoName))
    else :
        # rename this no-audio video
        os.system("mv {} {}".format(cacheVideoName, output))
    print("finish remix")
    print("done")


def doWithImage(src, output = 'out.jpg'):
    print("progressing")
    img = cv2.imread('test.jpg')
    vhs = VHSImage(img)
    cv2.imwrite(output, vhs)
    print("done")

if __name__ == "__main__":
    # doWithImage('test.jpg')
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type = str, required = False, default = '/Users/dabby/Desktop/视频/93539180-1-80.mp4')
    parser.add_argument("-o", "--output", type = str, default = "/Users/dabby/Desktop/output.mp4")
    args = vars(parser.parse_args())

    doWithVideo(args["input"], output = args["output"])