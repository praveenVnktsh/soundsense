import cv2
import numpy as np
import cv2.aruco as aruco
import pyrealsense2 as rs
import wave
import pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 8192
WAVE_OUTPUT_FILENAME = "file.wav"


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash
class ArucoDetector:

    def __init__(self) -> None:
        # Define the marker size in meters
        self.marker_size = 0.1

        # Initialize the RealSense D435i camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # Wait for a frameset and retrieve the camera intrinsics
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # Get the camera matrix
        self.camera_matrix = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],
                                  [0, self.intrinsics.fy, self.intrinsics.ppy],
                                  [0, 0, 1]])

        self.distortion = np.array(self.intrinsics.coeffs)
        self.cap = cv2.VideoCapture(8)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK,
                    input_device_index=5
            )
        
        self.audio_frames = []

        
    def capture_image(self):
        ret, img = self.cap.read()
        # Capture a single frame from the camera
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        cv2.imshow('frame', image)
        cv2.imshow('frame2', img)
        data = self.stream.read(CHUNK)
        self.audio_frames.append(data)
        return image

    def detect_aruco(self, image):
        # Detect all ArUco markers in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        # aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        # parameters = aruco.DetectorParameters_create()
        # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, rejectedImgPoints = detector.detectMarkers(gray,)
        
        if ids is not None:
            # ArUco markers detected
            # Estimate the pose of each detected marker
            # TODO distortion coefficients are hardcoded to 0
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            aid = 2
            if aid in ids:
                idx = list(ids).index(aid)
                rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.distortion)
                rvec, tvec = rvecs[idx], tvecs[idx]
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                translation_vector = tvec.reshape((3, 1))
                self.camera_pose = np.hstack((rotation_matrix, translation_vector))
                # get rectangle:
                top_left = corners[idx][0][0]
                top_left[0] += 10
                top_left[1] += 10
                bottom_right = corners[idx][0][2]
                bottom_right[0] -= 10
                bottom_right[1] -= 10
                top_left = map(int, top_left)
                bottom_right = map(int, bottom_right)
                top_left = tuple(top_left)
                bottom_right = tuple(bottom_right)
                crop = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                # cv2.imshow('crop', crop[:, :, 1])
                print(np.mean(crop[:, :, 1], axis=(0, 1)))
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        else:
            self.camera_pose = None
        cv2.imshow('frame', image)

    def __del__(self):
        # Stop the RealSense D435i camera
        self.pipeline.stop()


if __name__ == "__main__":
    # Create an instance of the ArucoDetector class
    aruco_detector = ArucoDetector()

    while True:
        # Call the capture_image method
        image = aruco_detector.capture_image()

        # Call the detect_aruco method
        aruco_detector.detect_aruco(image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            aruco_detector.stream.stop_stream()
            aruco_detector.stream.close()
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(aruco_detector.audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            print(len(aruco_detector.audio_frames))
            waveFile.writeframes(b''.join(aruco_detector.audio_frames))
            waveFile.close()
            break

