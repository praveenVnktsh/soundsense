import cv2
import numpy as np
import json
# Specify the ArUco dictionary to use
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Set up your video source (webcam or video file)
cap = cv2.VideoCapture('/home/praveen/dev/mmml/soundsense/data/run2/video.MP4')  # Use 0 for webcam
    
with open('data/calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

mtx, dist = calibration_data['mtx'], calibration_data['dist']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
    if not ret:
        break

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray, )
    print(ids)
    if ids is not None:  # Check if markers were found
        # Estimate pose of each marker
        # rvecs, tvecs, _ = detector.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)  # Adjust marker size as needed
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # for i in range(len(ids)):
        #     cv2.aruco.drawAxis(frame, mtx, dist, rvecs[i], tvecs[i], 0.03)  # Adjust axis length as needed

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
