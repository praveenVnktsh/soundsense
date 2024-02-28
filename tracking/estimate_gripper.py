import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
# Specify the ArUco dictionary to use


# Set up your video source (webcam or video file)
cap = cv2.VideoCapture('/home/praveen/dev/mmml/soundsense/data/videos/3.mp4')  # Use 0 for webcam
    
with open('realsense_calib.json', 'r') as f:
    calibration_data = json.load(f)

mtx, dist = calibration_data['mtx'], calibration_data['dist']
mtx = np.array(mtx)
dist = np.array(dist)
# 3 is the id of the camear
# 0 is the id of hte left gripper
# 1 is the id of the right gripper.
# 2 is the global coordinate frame.
ids_care = [0, 1, 2, 3]
relpose = {
    'lift' : 0.0,
    'extension' : 0.0,
    'lateral': 0.0,
    'roll' : 0.0,
    'gripper' : 0.0,
}
calib_frame = np.eye(4)
k = 0
thresholds = {
    'yaw' : 0.2
}
actions = [
    # lift, extension, lateral, roll, gripper
    (0, 0, 0, 0, 0)
]


avg_tvecs = []
rpys = []
prevgripper = 0
ground_frame = np.eye(4)

lframes = []
rframes = []
poses = {
    0 : [],
    1 : [],
    2 : [],
    3 : []
}
is_track = {
    0 : (True, 0),
    1 : (True, 0),
    2 : (True, 0),
    3 : (True, 0)
}
frame_idx = 0
filters = {
    0 : cv2.KalmanFilter(18, 6),
    1 : cv2.KalmanFilter(18, 6),
    2 : cv2.KalmanFilter(18, 6),
    3 : cv2.KalmanFilter(18, 6)

}
viz_frames = []
print("TRACKING PHASE...")
nparticles = 300
rthresh = 30
lcenter = np.array([825, 525])
rcenter = np.array([523, 525])
old_points_left = [lcenter + np.random.rand(2) * rthresh for i in range(100)]
old_points_left = np.array(old_points_left, dtype=np.float32)

old_points_right = [rcenter + np.random.rand(2) * rthresh for i in range(100)]
old_points_right = np.array(old_points_right, dtype=np.float32)

parameter_lucas_kanade = dict(winSize=(30, 30), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.02))
ret, frame = cap.read()
frame = frame[:, 1280:]
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
actions = []
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    rframe = frame[:, 1280:]
    rframes.append(rframe)
    gray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    viz_frames.append(cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR))
    
    frame_gray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('frame_gray', frame_gray)
    # cv2.waitKey(1)
    new_points_left, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, old_points_left, None,
                                                        **parameter_lucas_kanade)
    
    new_points_right, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, old_points_right, None,
                                                        **parameter_lucas_kanade)
    for point in new_points_left:
        cv2.circle(rframe, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    for point in new_points_right:
        cv2.circle(rframe, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    frame_gray_init = frame_gray.copy()
    old_points_left = []
    
    lmean = np.mean(new_points_left, axis=0)
    for pt in new_points_left:
        if np.linalg.norm(pt - lmean) < rthresh:
            old_points_left.append(pt)
        else:
            old_points_left.append(lcenter + np.random.randn(2) * 10)
    old_points_left = np.array(old_points_left, dtype=np.float32)
    old_points_right = []
    rmean = np.mean(new_points_right, axis=0)
    for pt in new_points_right:
        if np.linalg.norm(pt - rmean) < rthresh:
            old_points_right.append(pt)
        else:
            old_points_right.append(rcenter + np.random.randn(2) * 10)
    old_points_right = np.array(old_points_right, dtype=np.float32)
    
    cv2.circle(rframe, rmean.astype(int), 5, (0, 0, 255), -1)
    cv2.circle(rframe, lmean.astype(int), 5, (0, 0, 255), -1)
    dist = np.linalg.norm(rmean - lmean)
    if dist > 300:
        actions.append(0)
        cv2.arrowedLine(rframe, (640, 360), (540, 360), (255, 255, 0), 2)
        cv2.arrowedLine(rframe, (640, 360), (740, 360), (255, 255, 0), 2)
    else:
        actions.append(1)
        cv2.arrowedLine(rframe, (540, 360), (640, 360) , (0, 0, 255), 2)
        cv2.arrowedLine(rframe,  (740, 360), (640, 360), (0, 0, 255), 2)


    print(dist)
    cv2.imshow('frame', rframe)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

with open('actions.json', 'w') as f:
    json.dump(actions, f)

# Release resources
cap.release()
cv2.destroyAllWindows()


# TODO
# 1. Add gripper width estimation 
# 2. Handle loss of tracking using redundancy estimation and interpolation / backfilling of frames. (each frame must have an action estimate for each dimenision as relative pose.)
# 3. 
