import numpy as np

import cv2

import os
import glob

for path in glob.glob('frames/*.png'):
    img = cv2.imread(path)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (1920, 1080))
    cv2.imwrite(path, img)