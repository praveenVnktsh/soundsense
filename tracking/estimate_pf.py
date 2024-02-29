import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
# Specify the ArUco dictionary to use

class ParticleFilter():

    def __init__(self, centroid, num_particles = 100):
        self.particles = centroid + np.random.randn(num_particles, 2) * 20
        self.particles = np.array(self.particles, dtype=np.float32)
        self.parameter_lucas_kanade = dict(winSize=(30, 30), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.02))

    def update(self, frame, prev_frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        old_points = self.particles
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, old_points, None,
                                                            **self.parameter_lucas_kanade)
        mean = np.mean(new_points, axis=0)
        new_particles = []
        for particle in new_points:
            if np.linalg.norm(particle - mean) < 30:
                new_particles.append(particle)
            else:
                new_particles.append(mean + np.random.randn( 2) * 20)
        self.particles = np.array(new_particles, dtype=np.float32)

    def estimate(self):
        return np.mean(self.particles, axis=0)

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
    start_frame = is_track[aruco_id][1]
    packet = poses[aruco_id][start_frame - 1]
    # if 'corners' not in packet:
    #     mean = packet['particle_mean']
    # else:
    corners = packet['corners'].squeeze()
    mean = corners.mean(axis=0)
    # print(corners.shape)
    height = np.linalg.norm(corners[0] - corners[1])
    width = np.linalg.norm(corners[0] - corners[3])

    pf = ParticleFilter(mean)
    for i in range(start_frame, len(frames)):
        frame = frames[i]
        prev_frame = frames[i - 1]
        pf.update(frame, prev_frame)
        est = pf.estimate().squeeze()
        corners = np.array([
                [est[0] - width / 2, est[1] - height / 2],
                [est[0] - width / 2, est[1] + height / 2],
                [est[0] + width / 2, est[1] + height / 2],
                [est[0] + width / 2, est[1] - height / 2],
            ])
        rvecs, tvecs, mats = my_estimatePoseSingleMarkers([corners], 0.025, mtx, dist)
        poses[aruco_id][i] = {
            'particle_mean' : est.astype(np.int32),
            'corners' : corners,
            'rvec' : rvecs[0],
            'tvec' : tvecs[0],
        }

        viz_frame = viz_frames[i]
        for particle in pf.particles:
            particle = particle.squeeze()
            cv2.circle(viz_frame, (int(particle[0]), int(particle[1])), 2, (0, 255, 0), -1)
        cv2.circle(viz_frame, (int(est[0]), int(est[1])), 5, (0, 0, 255), -1)

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
        center = np.mean(corners[i].squeeze(), axis=0)
        poses[aruco_id][-1] = {
            'rvec' : rvecs[i],
            'tvec' : tvecs[i],
            'mat' : mats[i],
            'id_in_ground_frame' : id_in_ground_frame,
            'corners' : corners[i],
            'particle_mean' : center.astype(np.int32)
        }
        cv2.circle(viz_frames[-1], (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
    
    for aruco_id in range(4):
        if is_track[aruco_id][0] == False:
            if cur_frame_track[aruco_id] == True:
                last_seen_idx = is_track[aruco_id][1]
                # cur_tvec = poses[aruco_id][-1]['tvec']
                # prev_seen_tvec = poses[aruco_id][last_seen_idx]['tvec']
                # cur_rvec = poses[aruco_id][-1]['rvec']
                # prev_seen_rvec = poses[aruco_id][last_seen_idx]['rvec']
                
                poses = track_and_reestimate(rframes, poses, is_track, frame_idx, aruco_id, viz_frames)
    for k, v in cur_frame_track.items():
        if v == False:
            is_track[k] = (False, is_track[k][1])
        else:
            is_track[k] = (True, frame_idx)

    frame_idx += 1
print("REL POSE PHASE")



def get_slope(i):
    lgripper = poses[0][i]['particle_mean']
    rgripper = poses[1][i]['particle_mean']

    slope = (rgripper[1] - lgripper[1]) / (rgripper[0] - lgripper[0])
    angle = np.arctan(slope)
    return angle


for i in range(1, len(lframes)):
    for aruco_id in range(4):
        # if poses[aruco_id][i] is None:
        #     continue
        center = poses[aruco_id][i]['particle_mean']
        cv2.circle(viz_frames[i], (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        cv2.drawFrameAxes(viz_frames[i], mtx, dist, poses[aruco_id][i]['rvec'], poses[aruco_id][i]['tvec'], 0.025)
    
    angle_cur = get_slope(i)
    angle_prev = get_slope(i - 10)
    delta = angle_cur - angle_prev

    cv2.imshow('frame', viz_frames[i])
    
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
