import numpy as np
import cv2
import glob
import json
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
h = 5
w = 8
load = False

num_points = 500

with open(f'data/calibration_data copy.json', 'r') as f:
    dic = {
        # "mtx": mtx,
        # "dist": dist,
        # "rvecs": rvecs,
        # "tvecs": tvecs
    }
    dic = json.load(f)
    mtx = np.array(dic["mtx"])
    dist = np.array(dic["dist"])
    rvecs = np.array(dic["rvecs"])
    tvecs = np.array(dic["tvecs"])

img = cv2.imread('/home/praveen/dev/mmml/soundsense/data/calib/030166.png')
# img = cv2.resize(img, (720, 1280))
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('data/calibresult_0.png', img)
mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, mtx)
# fisheye undistort:

# K = mtx
# D = dist
# DIM = (720, 1280)
# dim1 = img.shape[:2][::-1]  
# scaled_K = K * dim1[0] / DIM[0]  
# balance = 0
# new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim1, np.eye(3), balance=balance)
# # map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
# # dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
# dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    



# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.rectangle(dst, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0,255,255), 5)
# dst = cv2.resize(dst, img.shape[:2][::-1])
# new_img = np.hstack((img, dst))
print(dst.shape)

cv2.imwrite(f'data/calibresult_{num_points}.png', dst)