# pyyy

from cv2 import cv2
import numpy as np
import os
import time
import argparse

def MyStyle(src):
    copy = src.copy()
    # blur
    blursize = 3
    copy = cv2.GaussianBlur(copy, (blursize, blursize), 0)
    # sharpen 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    copy = cv2.filter2D(copy, -1, kernel)

    # change Saturation
    HSV = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    S = HSV[:, :, 1]
    S[:, :] = S[:, :] * 0.8
    copy = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    # translate color's channel
    move = 1
    cols = copy.shape[1]
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
    B[:, :] = B[:, :] * 0.94
    G = copy[:, :, 1]
    G[:, :] = G[:, :] * 0.99
    return copy

def VXStyle(src):
    copy = src.copy()

    # make more yellow
    B = copy[:, :, 0]
    B[:, :] = B[:, :] * 0.9

    # vx ?
    # blur
    blursize = 5
    copy = cv2.GaussianBlur(copy, (blursize, blursize), 0)

    # change Saturation
    HSV = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    S = HSV[:, :, 1]
    S[:, :] = S[:, :] * 0.65
    # make brighter
    Vfloat = HSV[:, :, 2].astype(np.int)
    Vfloat[:] = Vfloat[:] * 1.5
    Vfloat[Vfloat > 255] = 255
    HSV[:, :, 2] = Vfloat

    copy = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

    return copy

def HandleImage(src):
    # return MyStyle(src)
    return VXStyle(src)    

def doWithVideo(src, output = 'out.mp4', encodewith264 = True, framepersecond = 30, interlaced = True, perferheight = 720):
    cap = cv2.VideoCapture(src)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total frame: {}'.format(framecount))

    if not os.path.exists(src):
        print('source does not exist')
        return False

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

    cachePrefixName = time.strftime('%Y%m%d_%H%M%S')
    cacheVideoName = "{}.cachevideo.mp4".format(cachePrefixName)

    cc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cacheVideoName, cc, fps, storesize)
    if out.isOpened() == False:
        print('fail to open video writer')
        return False

    currentframeindex = 0
    lastframe = np.array([])
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, videosize)
        # crop frame into storesize
        if shouldCropCols > 0:
            frame = frame[:, shouldCropCols: shouldCropCols + storesize[0]]
        if shouldCropRows > 0:
            frame = frame[shouldCropRows: shouldCropRows + storesize[1]]
        frame = HandleImage(frame)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        if interlaced:
            if lastframe.any():
                copyframe = frame.copy()
                copyframe[::2] = lastframe[::2]
                out.write(copyframe)
            else:
                copyframe = frame.copy()
                copyframe[::2] = copyframe[1::2]
                out.write(copyframe)
            lastframe = frame
        else:
            out.write(frame)
        print('progress:{}/{}'.format(currentframeindex, framecount))
        currentframeindex += 1
    cap.release()
    out.release()
    print('finish progress')

    print('remix audio and video')
    
    # ffmpeg -i INPUT_Video -i INPUT_Audio -map 0:v -map 1:a -c:v copy -c:a copy output.mp4
    ffmpegcommand = "ffmpeg -i {} -i {} -map 0:v -map 1:a -c:v {} -c:a copy {} {}".format(cacheVideoName, src, 'libx264' if encodewith264 else 'copy', '-r {}'.format(framepersecond) if (framepersecond > 0 and encodewith264) else '', output)
    print('command: {}'.format(ffmpegcommand))
    os.system(ffmpegcommand)
    print("cleaning cache")
    os.system("rm -rf {}".format(cacheVideoName))

    print("finish remix")
    print("done")


def doWithImage(src, output = 'out.jpg'):
    print("progressing")
    img = cv2.imread(src)
    row = img.shape[0]
    col = img.shape[1]
    width = 360
    height = int(width / col * row)
    img = cv2.resize(img, (width, height))
    vhs = HandleImage(img)
    # cv2.imwrite(output, vhs)
    cv2.imshow('output', vhs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")

if __name__ == "__main__":
    # doWithImage('test.jpg')
    
    username = "jam"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type = str, default = '/Users/{}/Desktop/input.mp4'.format(username))
    parser.add_argument("-o", "--output", type = str, default = "/Users/{}/Desktop/output.mp4".format(username))
    parser.add_argument("-height", "--perferheight", type = int, default = 480)
    parser.add_argument("-fps", "--framepersecond", type = int, default = 0)
    parser.add_argument("-x264", "--encode264", type = int, default = 1)
    parser.add_argument("-interlaced", "--interlaced", type = int, default = 1)
    args = vars(parser.parse_args())

    print(args)

    doWithVideo(args["input"], output = args["output"], encodewith264 = (args["encode264"] > 0), perferheight = args["perferheight"], framepersecond = args["framepersecond"], interlaced = args["interlaced"])