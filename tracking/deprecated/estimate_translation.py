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


def track_and_reestimate(frames, poses, is_track, frame_idx, aruco_id, viz_frames):
    
    parameter_lucas_kanade = dict(winSize=(30, 30), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.02))
    old_points = poses[aruco_id][is_track[aruco_id][1]]['corners'].squeeze().tolist()
    old_points.append(np.mean(old_points, axis=0))
    old_points = np.array(old_points, dtype=np.float32)
    

    frame_gray_init = cv2.cvtColor(frames[is_track[aruco_id][1]], cv2.COLOR_BGR2GRAY)

    for j in range(is_track[aruco_id][1], frame_idx):
        
        frame_gray = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
        
        # cv2.imshow('frame_gray', frame_gray)
        # cv2.waitKey(1)
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, old_points, None,
                                                            **parameter_lucas_kanade)
        for point in new_points:
            cv2.circle(viz_frames[j], (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        new_points = new_points[:4]
        # print(len(new_points), len(old_points))
        frame_gray_init = frame_gray.copy()
        old_points = new_points.tolist()
        old_points.append(np.mean(old_points, axis=0))
        old_points = np.array(old_points, dtype=np.float32)

        rvec, tvec, mat = my_estimatePoseSingleMarkers([new_points], 0.025, mtx, dist)
        rvec = rvec[0]
        tvec = tvec[0]
        mat = mat[0]

        mat = np.eye(4)
        rot_mat = cv2.Rodrigues(rvec)[0]
        mat[:3, :3] = rot_mat
        mat[:3, 3] = tvec.squeeze()
        poses[aruco_id][j] = {
            'rvec' : rvec,
            'tvec' : tvec,
            'mat' : mat,
            'id_in_ground_frame' : np.linalg.inv(ground_frame) @ mat
        }
        

    return poses



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
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# parameters.cornerRefinementMaxIterations = 100
# parameters.cornerRefinementMinAccuracy = 0.01
# parameters.cornerRefinementWinSize = 5
# parameters.adaptiveThreshWinSizeMin = 3
# parameters.adaptiveThreshWinSizeMax = 23
# parameters.adaptiveThreshWinSizeStep = 10
# parameters.adaptiveThreshConstant = 7
# parameters.minMarkerPerimeterRate = 0.05
# parameters.maxMarkerPerimeterRate = 4.0
# parameters.polygonalApproxAccuracyRate = 0.05
# parameters.minCornerDistanceRate = 0.05
# parameters.minDistanceToBorder = 3
# parameters.minMarkerDistanceRate = 0.05
# parameters.maxErroneousBitsInBorderRate = 0.04
# parameters.errorCorrectionRate = 0.6
# parameters.doCornerRefinement = True




detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

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
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    rframe = frame[:, :1280]
    lframe = frame[:, 1280:]
    lframes.append(lframe)
    rframes.append(rframe)
    gray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    viz_frames.append(cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR))
    # gray = cv2.medianBlur(gray, 3)
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
        if aruco_id == 2 and frame_idx == 0:
            ground_frame = mats[i]
        id_in_ground_frame = np.linalg.inv(ground_frame) @ mats[i]
        poses[aruco_id][-1] = {
            'rvec' : rvecs[i],
            'tvec' : tvecs[i],
            'mat' : mats[i],
            'id_in_ground_frame' : id_in_ground_frame,
            'corners' : corners[i]
        }
    
    for aruco_id in range(4):
        if is_track[aruco_id][0] == False:
            if cur_frame_track[aruco_id] == True:
                last_seen_idx = is_track[aruco_id][1]
                cur_tvec = poses[aruco_id][-1]['tvec']
                prev_seen_tvec = poses[aruco_id][last_seen_idx]['tvec']
                cur_rvec = poses[aruco_id][-1]['rvec']
                prev_seen_rvec = poses[aruco_id][last_seen_idx]['rvec']
                if aruco_id == 3:
                    poses = track_and_reestimate(rframes, poses, is_track, frame_idx, aruco_id, viz_frames)
    for k, v in cur_frame_track.items():
        if v == False:
            is_track[k] = (False, is_track[k][1])
        else:
            is_track[k] = (True, frame_idx)
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) == ord('q'):
    #     exit()
    frame_idx += 1

print("RELATIVE TRANSLATION ESTIMATION PHASE...")
# delx, dely, delz, delroll, delgripper = 0, 0, 0, 0, 0
actions = [[0, 0, 0, 0, 0]]
cumdelta = np.array([0, 0, 0], dtype=np.float32)
cumdelta = poses[3][0]['id_in_ground_frame'][:3, 3]
indices = [i for i in range(1, len(rframes), 10)]
for idx in range(1, len(indices)):
    i = indices[idx]
    mat = poses[3][indices[idx]]['id_in_ground_frame'] @ rigid_transform_to_center
    prev_mat = poses[3][indices[idx - 1]]['id_in_ground_frame'] @ rigid_transform_to_center
    cur_tvec = mat[:3, 3]
    prev_tvec = prev_mat[:3, 3]
    delta_tvec = cur_tvec - prev_tvec
    cumdelta += delta_tvec
    print("delta between frame", indices[idx], "and frame", indices[idx - 1], "is:")
    print(cur_tvec)
    print(prev_tvec)
    print(delta_tvec)
    actions.append([delta_tvec[2], delta_tvec[0], delta_tvec[1], 0, 0])
    delx, delz, dely = delta_tvec

    center = (640, 360)
    if abs(dely) > 0.0:
        length = int(100 * abs(delz) / 0.1)
        if dely < 0:
            cv2.arrowedLine(lframes[i], center, (center[0], center[1] + length) , (0, 255, 0), 2)
        else:
            cv2.arrowedLine(lframes[i],  center, (center[0], center[1] - length),(0, 255, 0), 2)

    if abs(delz) > 0.0:
        col = int (255 * abs(dely) / 0.1 + 128)
        if delz < 0:
            cv2.circle(lframes[i], center, 30, (0, col, 0), -1)
        else:
            cv2.circle(lframes[i], center, 30, (0, 0, col), -1)
            

    if abs(delx) > 0.0:
        length = int(100 * abs(delx) / 0.1)
        if delx < 0:
            cv2.arrowedLine(lframes[i], center, (center[0] - length, center[1]) , (0, 0, 255), 2)
        else:
            cv2.arrowedLine(lframes[i],  center, (center[0] + length, center[1]),(0, 0, 255), 2)

# for i in range(len(lframes)):
    for aruco_id in range(4):
        if poses[aruco_id][i] is not None:
            rvec = poses[aruco_id][i]['rvec']
            tvec = poses[aruco_id][i]['tvec']
        if rvec is  None or tvec is  None:
            continue
        rframe = viz_frames[i]
        lframe = lframes[i]
        cv2.drawFrameAxes(rframe, mtx, dist, rvec, tvec, length=0.03)

    center_link =  poses[aruco_id][i]['mat'] @ rigid_transform_to_center
    rvec = R.from_matrix(center_link[:3, :3]).as_rotvec()
    tvec_in_cam = center_link[:3, 3]
    cv2.drawFrameAxes(rframe, mtx, dist, rvec, tvec_in_cam, length=0.03 )
    
    fr = np.hstack([rframe, lframe])
    fr = cv2.resize(fr, (0, 0), fx = 0.75, fy = 0.75)
    cv2.imshow('frame', fr)
    
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
