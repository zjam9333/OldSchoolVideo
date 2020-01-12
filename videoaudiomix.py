# coding:utf-8
import os

def mixVideoAudio(videosrc, audiosrc, output, fps = 30):
    print('Remix audio and video')
    
    # remove " " before express command
    videosrc = videosrc.replace(" ", "\ ")
    audiosrc = audiosrc.replace(" ", "\ ")
    output = output.replace(" ", "\ ")
    # ffmpeg -i INPUT_Video -i INPUT_Audio -map 0:v -map 1:a -c:v copy -c:a copy output.mp4
    # ffmpegcommand = "ffmpeg -i {} -i {} -map 0:v -map 1:a -c:v {} -c:a copy {} {}".format(cacheVideoName, src, 'libx264' if encodewith264 else 'copy', '-r {}'.format(framepersecond) if (framepersecond > 0 and encodewith264) else '', output)
    ffmpegcommand = "ffmpeg -i {} -i {} -map 0:v -map 1:a -c:v copy {} {}".format(videosrc, audiosrc, '-r {}'.format(fps), output)
    print('FFMpeg command: {}'.format(ffmpegcommand))
    os.system(ffmpegcommand)