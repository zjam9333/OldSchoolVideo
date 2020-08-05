# coding:utf-8
from cv2 import cv2
import numpy as np

fps = 60
width = 1920 #3840
height = 1080 #2160
outputfile = "/Users/jam/Desktop/60fps.mp4"
coder = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(outputfile, coder, fps, (width, height))

blackframe = np.zeros((height, width, 3), np.uint8)
for i in range(120):
    videowriter.write(blackframe)
    print(i, 120)
videowriter.release()