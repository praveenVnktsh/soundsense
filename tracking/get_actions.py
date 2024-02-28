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


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Set up your video source (webcam or video file)
cap = cv2.VideoCapture('/home/praveen/dev/mmml/soundsense/data/videos/1.mp4')  # Use 0 for webcam
    
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
while True:
    ret, frame = cap.read()
    k += 1
    if k % 1 != 0:
        continue
    if not ret:
        break
    rframe = frame[:, :1280]
    lframe = frame[:, 1280:]
    # frame = cv2.resize(frame, (1920, 1080))
    gray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None:  # Check if markers were found
        # if k == 1:
        rvecs, tvecs, mats = my_estimatePoseSingleMarkers(corners, 0.025, mtx, dist)
        print(len(ids), len(rvecs))
        for i in range(len(ids)):
            aruco_id = ids[i][0]
            if aruco_id == 2:
                ground_frame = mats[i]
            try:
                id_in_ground_frame = np.linalg.inv(ground_frame) @ mats[i]
            except:
                print(aruco_id)
            if aruco_id == 3:
                # print(id_in_ground_frame)
                tvec = id_in_ground_frame[:3, 3]
                rpy = R.from_matrix(id_in_ground_frame[:3, :3]).as_euler('xyz')
                tvec = np.round(tvec, 3)
                print(tvec)


        cv2.aruco.drawDetectedMarkers(rframe, corners, ids)

    cv2.imshow('fram1e', lframe)
    cv2.imshow('frame', rframe)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

with open('actions.json', 'w') as f:
    json.dump(actions, f)

# Release resources
cap.release()
cv2.destroyAllWindows()
