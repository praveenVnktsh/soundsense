

import cv2
cap = cv2.VideoCapture('/dev/video7', cv2.CAP_V4L2)
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