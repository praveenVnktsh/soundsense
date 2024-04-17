import numpy as np
import json
import glob

cumsum = np.zeros((11, ))
for path in glob.glob('/home/punygod_admin/SoundSense/soundsense/data/mulsa/data/*/actions.json'):
    with open(path, 'r') as f:
        actions = np.array(json.load(f))
        actions = np.sum(actions, axis=0)
        cumsum += actions

print(cumsum)
weights = 1/cumsum
weights = weights/np.sum(weights)
print(weights.tolist())