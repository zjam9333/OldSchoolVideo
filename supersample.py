from cv2 import cv2
import numpy as np 

def jumpSample1080(imgin4k):
    something = imgin4k[0::2, 0::2].copy()
    return something

def superSample1080(imgin4k):
    m = imgin4k
    l1 = m[0::2]
    l2 = m[1::2]
    m = cv2.addWeighted(l1, 0.5, l2, 0.5, 1)
    l1 = m[:, 0::2]
    l2 = m[:, 1::2]
    m = cv2.addWeighted(l1, 0.5, l2, 0.5, 1)
    return m

if __name__ == "__main__":
    img4k = cv2.imread('4k.png')
    jump1080 = jumpSample1080(img4k)
    cv2.imwrite("jump1080.png", jump1080)

    super1080 = superSample1080(img4k)
    cv2.imwrite("super1080.png", super1080)