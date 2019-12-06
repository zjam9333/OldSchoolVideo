# coding:utf-8
import numpy as np
import cv2

# https://github.com/spmallick/learnopencv/blob/master/VideoStabilization/video_stabilization.py#L27

# The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS=20 

def movingAverage(curve, radius): 
    window_size = 2 * radius + 1
    # Define the filter 
    f = np.ones(window_size)/window_size 
    # Add padding to the boundaries 
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
    # Apply convolution 
    curve_smoothed = np.convolve(curve_pad, f, mode='same') 
    # Remove padding 
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed 

def smooth(trajectory): 
    smoothed_trajectory = np.copy(trajectory) 
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.2)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

videosrc = '/Users/zjj/Downloads/照片/C0371.mov'
cap = cv2.VideoCapture(videosrc)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read first frame
_, prev = cap.read() 
 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32) 

for i in range(n_frames - 2):
    ret, curr_frame = cap.read()
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    prev_pts = prev_pts[status == 1]
    curr_pts = curr_pts[status == 1]

    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    # Extract traslation
    dx = m[0,2]
    dy = m[1,2]

    # Extract rotation angle
    da = np.arctan2(m[1,0], m[0,0])
   
    # Store transformation
    transforms[i] = [dx,dy,da]

    prev_gray = curr_gray
    print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0) 
 
# Create variable to store smoothed trajectory
smoothed_trajectory = smooth(trajectory) 

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
 
# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
videocoder = cv2.VideoWriter_fourcc(*'mp4v')
cacheVideoName = 'testOutput.mp4'
videowriter = cv2.VideoWriter(cacheVideoName, videocoder, fps, (width, height))

# Write n_frames-1 transformed frames
for i in range(n_frames-2):
    # Read next frame
    success, frame = cap.read() 
    if not success:
        break

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (width, height))
    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized) 

    # Write the frame to the file
    # frame_stabilized = cv2.hconcat([frame, frame_stabilized])
    videowriter.write(frame_stabilized)
    
    print("writing: " + str(i) + "/" + str(n_frames))

cap.release()
videowriter.release()