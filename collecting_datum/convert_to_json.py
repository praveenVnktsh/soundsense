import numpy as np
import json
import os

# exit()
mapping = {
    'w': 0,
    's': 1,
    'a': 2,
    'd': 3,
    'n': 4,
    'm': 5,
    'i': 6,
    'k': 7,
    'j': 8,
    'l': 9,
}
# for folder
og_root = '/home/punygod_admin/SoundSense/soundsense/data/mulsa/data'
print(len(os.listdir(og_root)))
for run_id in os.listdir(og_root):
    root = f'{og_root}/{run_id}'
    print("Processing ", root)
    with open(f'{root}/keyboard_teleop.txt', 'r') as f:
        data = f.readlines()
        
        actions = []
        for line in data:
            line = line.split('\t')[1].strip()
            action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            if line in mapping.keys():
                action[mapping[line]] = 1
                actions.append(action)
            else:
                action[-1] = 1
                actions.append(action)

    with open(f'{root}/actions.json', 'w') as f:
        json.dump(actions, f)
