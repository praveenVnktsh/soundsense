import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
# Specify the ArUco dictionary to use

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
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
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


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
curpose = {
    0 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
    1 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
    2 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
    3 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
}

prevpose = {
    0 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
    1 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
    2 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
    3 : {'rvec' : [0, 0, 0], 'tvec' : [0, 0, 0]},
}

relpose = {
    'lift' : 0.0,
    'extension' : 0.0,
    'lateral': 0.0,
    'roll' : 0.0,
    'gripper' : 0.0,
}
calib_frame = np.eye(4)
transform_from_camera_to_gripper_center = np.eye(4)
k = 0
thresholds = {
    'yaw' : 0.1
}
actions = [
    # lift, extension, lateral, roll, gripper
    (0, 0, 0, 0, 0)
]
avg_tvecs = []
while True:
    ret, frame = cap.read()
    
    k += 1
    # if k % 5 != 0:
    #     continue
    if not ret:
        break
    rframe = frame[:, :1280]
    lframe = frame[:, 1280:]
    # frame = cv2.resize(frame, (1920, 1080))

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None:  # Check if markers were found
        # Estimate pose of each marker
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, 0.025, mtx, dist)
        cv2.aruco.drawDetectedMarkers(rframe, corners, ids)
        for i in range(len(ids)):
            id = ids[i][0]
            rvec = rvecs[i].squeeze()
            tvec = tvecs[i].squeeze()
            if id == 3: # camera_center
                # check if rotation
                curx, cury, curz = tvec
                # prevx, prevy, prevz = prevpose[id]['tvec']
                avg_tvecs.append(tvec)
                if len(avg_tvecs) > 20:
                    avg_tvecs.pop(0)
                else:
                    continue
                
                mean_tvec = np.mean(avg_tvecs, axis=0)
                prevx, prevy, prevz = mean_tvec

                delx = curx - prevx
                dely = cury - prevy
                delz = curz - prevz

                print(dely)
                # print(tvec)

                cur_rpy = R.from_rotvec(rvec).as_euler('zyx', degrees=False)
                prev_rpy = R.from_rotvec(prevpose[id]['rvec']).as_euler('zyx', degrees=False)
                cur_yaw = cur_rpy[0]
                prev_yaw = prev_rpy[0]
                prevpose[id]['rvec'] = rvec.copy()
                prevpose[id]['tvec'] = tvec.copy()

                if abs(cur_yaw - prev_yaw) > thresholds['yaw']:
                    action = (
                        0, 0, 0, cur_yaw - prev_yaw, 0
                    )
                    actions.append(action)
                    if cur_yaw > prev_yaw:
                        cv2.arrowedLine(lframe,  (400, 400), (500, 500),(0, 255, 255), 2)
                    else:
                        cv2.arrowedLine(lframe, (600, 500), (550, 400) , (0, 255, 255), 2)
                    continue
                if abs(delz) > 0.02:
                    action = (
                        curz - prevz, 0, 0, 0, 0
                    )
                    actions.append(action)
                    if delz > 0:
                        cv2.circle(lframe, (640, 360), 50, (0, 255, 0), 2)
                    else:
                        cv2.circle(lframe, (640, 360), 50, (0, 255, 0), -1)
                    continue
                if abs(delx) > 0.01:
                    action = (
                        0, 0, curx - prevx, 0, 0
                    )
                    actions.append(action)
                    if delx > 0:
                        cv2.arrowedLine(lframe,  (640, 360), (540, 360),(0, 255, 255), 2)
                    else:
                        cv2.arrowedLine(lframe, (640, 360), (740, 360) , (0, 255, 255), 2)
                    continue
                
                

                if abs(dely) > 0.01:
                    action = (
                        0, cury - prevy, 0, 0, 0
                    )
                    actions.append(action)
                    if dely > 0:
                        cv2.arrowedLine(lframe, (640, 360), (640, 460) , (0, 255, 255), 2)
                    else:
                        cv2.arrowedLine(lframe,  (640, 360), (640, 260),(0, 255, 255), 2)
                    continue
                    
                # print(delz)

            curpose[id] = {
                'rvec' : rvec,
                'tvec' : tvec 
            }

            relpose[id] = {
                'rvec' : rvec ,
                'tvec' : tvec 
            }
            
            # print(relpose)

        # rotation_matrix, _ = cv2.Rodrigues(rvec)
        # translation_vector = tvec.reshape((3, 1))
        # self.camera_pose = np.hstack((rotation_matrix, translation_vector))

    cv2.imshow('frame', rframe)
    cv2.imshow('fram1e', lframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
