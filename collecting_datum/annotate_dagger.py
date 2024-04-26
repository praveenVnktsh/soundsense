import numpy as np

import cv2
import glob
import os

root = '/home/punygod_admin/SoundSense/soundsense/data/mulsa/dagger_2'
for run_id in os.listdir(root):
    print(os.path.join(root, run_id, 'keyboard_teleop_new.txt'))
    f = open(os.path.join(root, run_id, 'keyboard_teleop_new.txt'), 'w')
    for path in sorted(glob.glob(os.path.join(root, run_id, 'video', '*.png'))):
        img = cv2.imread(path)
        cv2.imwrite('img.png', img)
        key = input()
        print(key)
        f.write(os.path.basename(path).split('.')[0] + ' ' + str(key) + '\n')
    f.close()
