import cv2
# check camera id in v4l2-ctl --list-devices
# resolution is 1280x720

cam = cv2.VideoCapture(6)
cv2.namedWindow("test")
print(cam.isOpened())


while True:
    ret, frame = cam.read()
    print(frame.shape)
    if not ret:
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))