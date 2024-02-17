import glob
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
dic = {
    "action_history" : [], 
    "camera_timestamp" : [],
    "audio_timestamp": []
    
}

root = '../data/'
for d in os.listdir(root):
    dirname = root + d  + '/'

    with open(dirname + 'CameraTrajectory.txt', 'r') as f:
        data = f.readlines()

    k = 0
    prevreading = np.array([0, 0, 0, 0, 0, 0, 0])
    for line in data:
        line = line.split(' ')
        frame_id = line[0]
        if k == 0:
            pose = ([0, 0, 0, 0, 0, 0])
            dic['camera_timestamp'].append(frame_id)
            dic['action_history'].append(pose)
            dic['audio_timestamp'].append(0)
            k += 1
            continue
        k += 1
        reading = list(map(float, line[1:]))
        xyz = np.array(reading[:3])
        quat = np.array(reading[3:])
        prevreading[:3] = xyz.copy()
        prevreading[3:] = quat.copy()
        pose = [
            *(xyz - prevreading[:3]),
            *(quat - prevreading[3:]),
        ]
        rpy = R.from_quat(pose[3:]).as_euler('zyx', degrees=True)

        dic['camera_timestamp'].append(frame_id)
        dic['action_history'].append([
            pose[0], 
            pose[1],
            pose[2],
            rpy[0],
            rpy[1],
            rpy[2]
        ])
        dic['audio_timestamp'].append(0)

        
    with open(dirname + 'ee_pose.json', 'w') as f:
        json.dump(dic, f)

    


    

