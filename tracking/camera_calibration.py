import numpy as np
import cv2
import glob
import json
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
h = 5
w = 8
load = False
# load = True
fisheye = False
# fisheye = True
num_points = 500
DIM = (1280, 720)
# DIM = (720, 1280)
if not load:
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((h*w,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('/home/praveen/dev/mmml/soundsense/data/calib2/*.png')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img.shape[0] < img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, DIM)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (h,w), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, (w,h), corners2, ret)
            cv2.imshow('img', cv2.resize(img, (0, 0), fx = 0.3, fy= 0.3))
            cv2.waitKey(1)
            print("Num good frames = ", len(objpoints), " / ", idx + 1)

        if len(objpoints) > num_points:
            break
    cv2.destroyAllWindows()
    np.save('data/calib_objpoints.npy', objpoints)
    np.save('data/calib_imgpoints.npy', imgpoints)
    np.save('data/calib_gray.npy', gray)
else:
    objpoints = np.load('data/calib_objpoints.npy')
    imgpoints = np.load('data/calib_imgpoints.npy')
    gray = np.load('data/calib_gray.npy')
    print("Loaded calibration data")


img = cv2.imread('/home/praveen/dev/mmml/soundsense/data/calib/030166.png')
# img = cv2.resize(img, (720, 1280))
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = cv2.resize(img, DIM)
cv2.imwrite('data/calibresult_0.png', img)





if fisheye:

    objpoints = np.array(objpoints[:num_points], dtype=np.float32)
    imgpoints = np.array(imgpoints[:num_points], dtype=np.float32)
    # dump the calibration data
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    calibration_flags = cv2.fisheye.CALIB_FIX_SKEW #+ cv2.CALIB_FIX_PRINCIPAL_POINT #+ cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

    objpoints = np.expand_dims(objpoints, -2)

    print(objpoints.shape, imgpoints.shape)


    rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )
    dic = {
            "mtx": K.tolist(),
            "dist": D.tolist(),
            "rvecs": np.array(rvecs).tolist(),
            "tvecs": np.array(tvecs).tolist()
        }
    
    dim1 = img.shape[:2][::-1]  
    scaled_K = K * dim1[0] / DIM[0]  
    balance = 0
    # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim1, np.eye(3), balance=balance)
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
    # dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
else: 

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        gray.shape[::-1], 
        None, 
        None,
        flags = cv2.CALIB_RATIONAL_MODEL
        )

    dic = {
            "mtx": mtx.tolist(),
            "dist": dist.tolist(),
            "rvecs": np.array(rvecs).tolist(),
            "tvecs": np.array(tvecs).tolist()
        }

    mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, mtx)

cv2.imwrite(f'data/calibresult.png', dst)
with open(f'data/calibration_data.json', 'w') as f:
    
    
    json.dump(dic, f)
