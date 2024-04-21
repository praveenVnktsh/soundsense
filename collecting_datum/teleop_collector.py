import stretch_body.robot as rb
import time
from pynput import keyboard
import datetime
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
keypress = None
def on_press(key):
    global keypress
    keypress = key

def on_release(key):
    global keypress
    keypress = None

class Collector():

    def __init__(self):
        self.robot = rb.Robot()
        self.robot.startup()
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
        self.stop = False
        self.cap =cv2.VideoCapture('/dev/video6', cv2.CAP_V4L2)
        self.file = open(Data_PATH + f'/{run_id}/keyboard_teleop.txt', 'w')
        # while not self.cap.isOpened():
        #     # self.cap = cv2.VideoCapture(camera_idx)
        #     print('Failed top open camera')
        #     exit()


    def reset(self):

        self.robot.arm.move_to(0.0)
        self.robot.lift.move_to(1.075)
        self.robot.end_of_arm.move_to('wrist_pitch', 0.0)
        self.robot.end_of_arm.move_to('wrist_yaw', 0.0)
        self.robot.end_of_arm.move_to('stretch_gripper', 100)
        self.robot.end_of_arm.move_to('wrist_roll', 0.0)
        self.robot.push_command()
        time.sleep(3)


    def get_command(self,):
        command = None
        

        c = str(keypress).replace("'", '')
        time = datetime.datetime.now().strftime("%s%f")
        delta_trans = 0.05
        delta_rad = np.pi * 6/ 180.0
        ret, frame = self.cap.read()
        self.file.write(time + '\t' + c + '\n')
        if frame is not None:
            pathFolder = Data_PATH + '/' + str(run_id) + '/' + 'video/' + str(time) + '.png'
            frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
            cv2.imwrite(pathFolder, frame)

        # print(time, c, pathFolder)
        
        if c is not None:
            print('---------- KEYBOARD TELEOP MENU -----------')
            print("""
                w - extend arm
                s - retract arm

                i - pitch up
                k - pitch down

                l - roll right
                j - roll left

                m - close gripper
                n - open gripper
            """)
        if c == 'w':
            self.robot.arm.move_by(delta_trans)
        elif c == 's':
            self.robot.arm.move_by(-delta_trans)

        elif c == 'i':
            self.robot.end_of_arm.move_by('wrist_pitch', delta_rad)
        elif c == 'k':
            self.robot.end_of_arm.move_by('wrist_pitch', -delta_rad)

        elif c == 'l':
            self.robot.end_of_arm.move_by('wrist_roll', delta_rad)
        elif c == 'j':
            self.robot.end_of_arm.move_by('wrist_roll', -delta_rad)

        elif c == 'n':
            self.robot.end_of_arm.move_by('stretch_gripper', 10)
        elif c == 'm':
            self.robot.end_of_arm.move_by('stretch_gripper', -10)
        
        elif c == 'q':
            self.stop = True
            pub = rospy.Publisher('end_recording', String, queue_size=10)
            pub.publish('end')
            self.file.close()
            return

        self.robot.push_command()


    def main(self):
        
        pub = rospy.Publisher('end_recording', String, queue_size=10)
        for i in range(2):
            pub.publish(f'start.{run_id}')
            time.sleep(1)
        while not self.stop:
            self.get_command()
            time.sleep(0.1)

        

if __name__ == '__main__':
    import os, sys
    rospy.init_node('temp')
    run_id = sys.argv[1]

    Data_PATH = '../data/sorting'
    os.makedirs(Data_PATH + f'/{run_id}/video', exist_ok=True)
    c = Collector()
    c.reset()

    c.main()
    print('Done')

        
    
        



