import numpy as np
import json
import glob
import cv2
import threading
import time
from playsound import playsound
# def alert():
#     threading.Thread(target=playsound, args=('/home/hello-robot/soundsense/soundsense/stretch/data/0/1711051394806730.wav',), daemon=True).start()

with open('data/0/keyboard_teleop.json', 'r') as f:
    data = json.load(f)

# https://drive.google.com/file/d/1WToAxRFNkrsGaALi2A5Fp9ZkiIL9N56Y/view?usp=sharing
paths = glob.glob('data/0/video/*.png')
print()
print(len(data), len(paths))

col = 255
# alert()
starttime = time.time()
for i, path in enumerate(sorted(glob.glob('data/0/video/*.png'))):
    
    img = cv2.imread(path)
    
    idx = data[i].index(1)

    
    length = 100
    center = (320, 160)
    

    
    if idx == 0:
        cv2.circle(img, center, 30, (0, col, 0), -1)
    elif idx == 1:
        cv2.circle(img, center, 30, (0, 0, col), -1)
        
    # if idx == 4:
    #     cv2.arrowedLine(img, center, (center[0], center[1] + length) , (0, 255, 0), 2)
    # elif idx == 5:
    #     cv2.arrowedLine(img,  center, (center[0], center[1] - length),(0, 255, 0), 2)

    

    if idx == 2:
        cv2.arrowedLine(img, center, (center[0] - length, center[1]) , (0, 0, 255), 2)
    elif idx == 3:
        cv2.arrowedLine(img,  center, (center[0] + length, center[1]),(0, 0, 255), 2)


    if idx == 4:
        cv2.putText(img, 'open', (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif idx == 5:
        cv2.putText(img, 'close', (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if idx == 6:
        cv2.arrowedLine(img, center, (center[0], center[1] - length) , (255, 0, 0), 2)
    elif idx == 7:
        cv2.arrowedLine(img,  center, (center[0], center[1] + length),(255, 0, 0), 2)



    if idx == 8:
        cv2.putText(img, 'roll left', (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif idx == 9:
        cv2.putText(img, 'roll right', (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('image', img)
    if cv2.waitKey(100) == ord('q'):
        break
print(time.time() - starttime)