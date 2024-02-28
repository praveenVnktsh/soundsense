import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
# Specify the ArUco dictionary to use

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion,):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    mats = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rot_mat = cv2.Rodrigues(R)[0]
        mat = np.eye(4)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = t.squeeze()
        mats.append(mat)
        rvecs.append(R)
        tvecs.append(t)
    return rvecs, tvecs, mats


rigid_transform_to_center = np.array(
    [
        [1, 0, 0, 0.025],
        [0, 1, 0, -0.07],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
)



aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Set up your video source (webcam or video file)
cap = cv2.VideoCapture('/home/praveen/dev/mmml/soundsense/data/videos/4.mp4')  # Use 0 for webcam
    
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
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    rframe = frame[:, :1280]
    lframe = frame[:, 1280:]
    lframes.append(lframe)
    rframes.append(rframe)
    gray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    rvecs, tvecs, mats = my_estimatePoseSingleMarkers(corners, 0.025, mtx, dist)
    for i in range(4):
        poses[i].append(None)

    cur_frame_track = {
        0 : False,
        1 : False,
        2 : False,
        3 : False
    }
    for i in range(len(ids)):
        aruco_id = ids[i][0]
        cur_frame_track[aruco_id] = True
        if aruco_id > 3:
            continue
        if aruco_id == 2:
            ground_frame = mats[i]
        id_in_ground_frame = np.linalg.inv(ground_frame) @ mats[i]
        poses[aruco_id][-1] = {
            'rvec' : rvecs[i],
            'tvec' : tvecs[i],
            'mat' : mats[i],
            'id_in_ground_frame' : id_in_ground_frame
        }
        cv2.drawFrameAxes(rframe, mtx, dist, rvecs[i], tvecs[i], length=0.03) 
    
    for aruco_id in range(4):
        if is_track[aruco_id][0] == False:
            cv2.putText(rframe, f'Lost tracking for {aruco_id}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if cur_frame_track[aruco_id] == True:
                cv2.putText(rframe, f'Recovered tracking for {aruco_id}', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # interpolate previous frames
                last_seen_idx = is_track[aruco_id][1]
                cur_tvec = poses[aruco_id][-1]['tvec']
                prev_seen_tvec = poses[aruco_id][last_seen_idx]['tvec']
                cur_rvec = poses[aruco_id][-1]['rvec']
                prev_seen_rvec = poses[aruco_id][last_seen_idx]['rvec']
                cv2.putText(rframe, f'Interpolating between {last_seen_idx} and {frame_idx}', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                for j in range(is_track[aruco_id][1], frame_idx):
                    
                    alpha = (j - last_seen_idx) / (frame_idx - last_seen_idx)
                    tvec = (1 - alpha) * prev_seen_tvec + alpha * cur_tvec
                    # rvec = (1 - alpha) * prev_seen_rvec + alpha * cur_rvec
                    rvec = cur_rvec
                    mat = np.eye(4)
                    rot_mat = cv2.Rodrigues(rvec)[0]
                    mat[:3, :3] = rot_mat
                    mat[:3, 3] = tvec.squeeze()
                    try:
                        poses[aruco_id][j] = {
                            'rvec' : rvec,
                            'tvec' : tvec,
                            'mat' : mat,
                            'id_in_ground_frame' : np.linalg.inv(ground_frame) @ mat
                        }
                    except:
                        print(j, last_seen_idx, frame_idx, aruco_id, poses.keys(), len(poses[aruco_id]))
                        exit()

                    cv2.drawFrameAxes(rframes[j], mtx, dist, rvec, tvec, length=0.03) 
                    cv2.imshow('frame1', rframes[j])
                    if cv2.waitKey(1) == ord('q'):
                        exit()
    for k, v in cur_frame_track.items():
        if v == False:
            is_track[k] = (False, is_track[k][1])
        else:
            is_track[k] = (True, frame_idx)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        exit()
    
        
    
    frame_idx += 1
        
        


rvec = None
tvec = None
for i in range(len(lframes)):
    if poses[3][i] is not None:
        rvec = poses[3][i]['rvec']
        tvec = poses[3][i]['tvec']
    if rvec is  None or tvec is  None:
        continue
    rframe = rframes[i]
    lframe = lframes[i]
    cv2.drawFrameAxes(rframe, mtx, dist, rvec, tvec, length=0.03) 
    fr = np.hstack([rframe, lframe])
    fr = cv2.resize(fr, (0, 0), fx = 0.75, fy = 0.75)
    cv2.imshow('frame', fr)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
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
