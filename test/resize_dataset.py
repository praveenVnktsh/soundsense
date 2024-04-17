import numpy as np
import cv2
import glob
H, W = 100, 75
for path in sorted(glob.glob('/home/punygod_admin/SoundSense/soundsense/data/mulsa/data_resized/*/video/*.png')):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (H, W))
    cv2.imwrite(path, img)