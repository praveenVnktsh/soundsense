import glob
import os

root = '/home/punygod_admin/SoundSense/soundsense/data/mulsa/sorting/'


for i, path in enumerate(sorted(glob.glob(root + '*/keyboard_teleop.txt'))):
   with open(path, 'r') as f:
    lines = f.readlines()
    lines = lines[5:]

    with open(path, 'w') as f:
        f.writelines(lines)

total = i
for j in range(total):
    for i, path in enumerate(sorted(glob.glob(root + f'{j}/video/*.png'))):
        if i < 5:
            os.remove(path)
            print(f"Removed {path}")
