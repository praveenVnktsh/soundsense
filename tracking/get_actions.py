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

center_link_poses = [
    [],
    []
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
        # print(len(ids), len(rvecs))
        for i in range(len(ids)):
            aruco_id = ids[i][0]
            if aruco_id == 2:
                ground_frame = mats[i]
            rvec = rvecs[i]
            tvec = tvecs[i]
            # cv2.drawFrameAxes(rframe, mtx, dist, rvec, tvec, length=0.03 )
            id_in_ground_frame = np.linalg.inv(ground_frame) @ mats[i]
            cv2.drawFrameAxes(rframe, mtx, dist, rvec, tvec, length=0.03 ) 
            if aruco_id == 3:
                center_link =  mats[i] @ rigid_transform_to_center
                rvec = R.from_matrix(center_link[:3, :3]).as_rotvec()
                tvec_in_cam = center_link[:3, 3]
                cv2.drawFrameAxes(rframe, mtx, dist, rvec, tvec_in_cam, length=0.03 )
                center_link_in_ground = np.linalg.inv(ground_frame) @ center_link
                tvec = center_link_in_ground[:3, 3]
                rvec = R.from_matrix(center_link_in_ground[:3, :3]).as_rotvec()
                # print(tvec)
                
                center_link_poses[0].append(tvec)
                center_link_poses[1].append(rvec)

                if len(center_link_poses[0]) > 10:
                    center_link_poses[0].pop(0)
                    center_link_poses[1].pop(0)
                else:
                    continue

                avg_tvec = np.mean(center_link_poses[0], axis=0)


                delp = tvec - avg_tvec
                delx, dely, delz = delp
                print(delx)

                center = (640, 360)
                if abs(delz) > 0.003:
                    length = int(100 * abs(delz) / 0.01)
                    
                    if delz < 0:
                        cv2.arrowedLine(lframe, center, (center[0], center[1] + length) , (0, 255, 0), 2)
                    else:
                        cv2.arrowedLine(lframe,  center, (center[0], center[1] - length),(0, 255, 0), 2)

                if abs(dely) > 0.003:
                    col = int (255 * abs(dely) / 0.1 + 128)
                    if dely < 0:
                        cv2.circle(lframe, center, 30, (0, col, 0), -1)
                    else:
                        cv2.circle(lframe, center, 30, (0, 0, col), -1)
                        

                if abs(delx) > 0.003:
                    length = int(100 * abs(delx) / 0.01)
                    if delx < 0:
                        cv2.arrowedLine(lframe, center, (center[0] - length, center[1]) , (0, 0, 255), 2)
                    else:
                        cv2.arrowedLine(lframe,  center, (center[0] + length, center[1]),(0, 0, 255), 2)


                rpy = R.from_rotvec(rvec).as_euler('zyx', degrees=False)
                prev_rpy = R.from_rotvec(center_link_poses[1][-2]).as_euler('zyx', degrees=False)
                cur_yaw = np.mod(rpy[0] + np.pi, 2 * np.pi) - np.pi
                prev_yaw = np.mod(prev_rpy[0] + np.pi, 2 * np.pi) - np.pi
                del_yaw = cur_yaw - prev_yaw
                if abs(del_yaw) > 0.01:
                    if del_yaw < 0:
                        cv2.ellipse(lframe, center, (100, 100), 0, 0, 80, (0, 0, 0), thickness=2)
                        cv2.arrowedLine(lframe, (740, 360), (740, 310), (0, 0, 0), thickness=2, tipLength=0.5)
                    else:
                        cv2.ellipse(lframe, center, (100, 100), 0, 0, 90, (0, 0, 0), thickness=2)
                        cv2.arrowedLine(lframe, (640, 460), (600, 460), (0, 0, 0), thickness=2, tipLength=0.5)

        cv2.aruco.drawDetectedMarkers(rframe, corners, ids)
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
