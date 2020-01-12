from cv2 import cv2
import numpy as np
import videoaudiomix
import argparse
from tqdm import tqdm
import os
import time

def progressVideo(src, output):
    if not os.path.exists(src):
        print('Input video path {} does not exist'.format(src))
        return False
    cap = cv2.VideoCapture(src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    sourcefps = 50
    if not fps == sourcefps:
        return False
    cacheNamePrefix = "cache_will_be_deleted_jowfajoasdfklajdsf"
    # clean old cache if need
    os.system("rm -rf {}*".format(cacheNamePrefix))
    cacheNameSuffix = time.strftime('%Y%m%d_%H%M%S')
    cacheVideoName = "{}_{}.mp4".format(cacheNamePrefix, cacheNameSuffix)

    print("counting source total frame")
    srctotalframe = 0
    while(cap.isOpened()):
        ret, _ = cap.read()
        if ret == True:
            srctotalframe += 1
        else:
            break
    print("source total frame = ", srctotalframe)
    targetfps = 30
    targettotalframe = float(srctotalframe) / float(sourcefps) * float(targetfps)
    targettotalframe = int(targettotalframe)

    videocoder = cv2.VideoWriter_fourcc(*'avc1')
    videowriter = cv2.VideoWriter(cacheVideoName, videocoder, targetfps, (width, height))
    if videowriter.isOpened() == False:
        print('Fail to open video writer')
        return False
    progressloop = tqdm(range(targettotalframe))
    for index in progressloop:
        # 新的帧是从原视频中抽两帧合并的
        positionfloat = float(index) / float(targetfps) * float(sourcefps)
        positionintfirst = int(positionfloat)
        secondweight = (positionfloat - float(positionintfirst))
        cap.set(cv2.CAP_PROP_POS_FRAMES, positionintfirst)
        _, frame1 = cap.read()
        _, frame2 = cap.read()
        thisframe = cv2.addWeighted(frame1, 1.0 - secondweight, frame2, secondweight, 0)
        videowriter.write(thisframe)
    
    progressloop.close()
    cap.release()
    videowriter.release()
    print('Finish progress')
    # mix video and audio
    videoaudiomix.mixVideoAudio(cacheVideoName, src, output)
    
    print("Cleaning cache")
    os.system("rm -rf {}*".format(cacheNamePrefix))

    print("Finish remix")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type = str, default = '/Users/jam/Desktop/test.mp4')
    parser.add_argument("-o", "--output", type = str, default = '/Users/jam/Desktop/test_out_30.MP4')
    userArgs = vars(parser.parse_args())
    progressVideo(userArgs["input"], output = userArgs["output"])

    