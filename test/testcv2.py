

import cv2
# cap = cv2.VideoCapture(f'v4l2src device=/dev/video7 io-mode=2 ! image/jpeg, width=(int)1920, height=(int)1080 !  appsink', cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture('/dev/video7', cv2.CAP_V4L2)
# cap = cv2.VideoCapture(6)
while True:

    ret, frame = cap.read()
    print(frame.shape)
    # print(ret)
    # break
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# print(cv2.getBuildInformation())