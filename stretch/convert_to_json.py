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
for run_id in os.listdir('hundred/data'):
    root = f'hundred/data/{run_id}'
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