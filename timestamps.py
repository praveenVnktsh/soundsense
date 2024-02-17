import json

with open('timestamps.json' ,'r') as f:
    dic = json.load(f)

print(dic.keys())
print(dic['cam_gripper_timestamps'][0])
print(dic['gelsight_timestamps'][0])
print(dic['audio_timestamps'][0])
print(dic['program_timestamps'][0])
print(dic['approach_end'][0])
print(dic['pose_history'][0])
print(dic['action_history'][0])