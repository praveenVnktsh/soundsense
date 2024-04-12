#!/usr/bin/env python3

import math
import keyboard as kb
import argparse as ap
import alsaaudio, wave, numpy

import rospy
from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import sys
import select
from pynput import keyboard
import datetime
from std_msgs.msg import String
import hello_helpers.hello_misc as hm
import cv2
import os


Data_PATH = '../stretch/data/data_two_cups'
keypress = None
def on_press(key):
    global keypress
    keypress = key

def on_release(key):
    global keypress
    keypress = None

class GetKeyboardCommands:

    def __init__(self, mapping_on, hello_world_on, open_drawer_on, clean_surface_on, grasp_object_on, deliver_object_on):
        self.mapping_on = mapping_on
        self.hello_world_on = hello_world_on
        self.open_drawer_on = open_drawer_on
        self.clean_surface_on = clean_surface_on
        self.grasp_object_on = grasp_object_on
        self.deliver_object_on = deliver_object_on

        self.step_size = 'big'
        self.rad_per_deg = math.pi/180.0
        self.small_deg = 3.0
        self.small_rad = self.rad_per_deg * self.small_deg
        self.small_translate = 0.005  #0.02
        self.medium_deg = 6.0
        self.medium_rad = self.rad_per_deg * self.medium_deg
        self.medium_translate = 0.04
        self.big_deg = 12.0
        self.big_rad = self.rad_per_deg * self.big_deg
        self.big_translate = 0.03
        self.mode = 'position' 
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
        os.makedirs(Data_PATH + f'/{run_id}/', exist_ok=True)
        self.file = open(Data_PATH + f'/{run_id}/keyboard_teleop.txt', 'w')
        # camera_idx = 
        # self.cap = cv2.VideoCapture(f'v4l2src device=/dev/video7 io-mode=2 ! image/jpeg, width=(int)1920, height=(int)1080 !  appsink', cv2.CAP_GSTREAMER)
        self.cap =cv2.VideoCapture('/dev/video6', cv2.CAP_V4L2)
        # self.cap = cv2.VideoCapture(camera_idx)
        while not self.cap.isOpened():
            # self.cap = cv2.VideoCapture(camera_idx)
            print('Failed top open camera')
            exit()

    def get_deltas(self):
        if self.step_size == 'small':
            deltas = {'rad': self.small_rad, 'translate': self.small_translate}
        if self.step_size == 'medium':
            deltas = {'rad': self.medium_rad, 'translate': self.medium_translate} 
        if self.step_size == 'big':
            deltas = {'rad': self.big_rad, 'translate': self.big_translate} 
        return deltas

    def print_commands(self, joint_state, command):
        if command is None:
            return

        joints = joint_state.name
        def in_joints(i):
            return len(list(set(i) & set(joints))) > 0

        print('---------- KEYBOARD TELEOP MENU -----------')
        print("""
            w - extend arm
            s - retract arm
            a - move left
            d - move right

            i - lift up
            k - lift down

            l - roll right
            j - roll left

            m - close gripper
            n - open gripper
        """)

    def get_command(self, node):
        command = None

        c = str(keypress).replace("'", '')
        time = datetime.datetime.now().strftime("%s%f")
        self.file.write(time + '\t' + c + '\n')
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)

        pathFolder = Data_PATH + '/' + str(run_id) + '/' + 'video/' + str(time) + '.png'
        # print(time, c, pathFolder)
        cv2.imwrite(pathFolder, frame)
        
        ####################################################
        ## MOSTLY MAPPING RELATED CAPABILITIES
        ## (There are non-mapping outliers.)
        ####################################################
        
        # Sequential performs a fixed number of autonomus mapping iterations
        if (c == '!') and self.mapping_on:
            number_iterations = 4
            for n in range(number_iterations):
                # Trigger a 3D scan with the D435i
                trigger_request = TriggerRequest() 
                trigger_result = node.trigger_head_scan_service(trigger_request)
                rospy.loginfo('trigger_result = {0}'.format(trigger_result))

                # Trigger driving the robot to the estimated next best place to scan
                trigger_request = TriggerRequest() 
                trigger_result = node.trigger_drive_to_scan_service(trigger_request)
                rospy.loginfo('trigger_result = {0}'.format(trigger_result))
                command = 'service_occurred'
                
        # Trigger localizing the robot to a new pose anywhere on the current map
        if ((c == '+') or (c == '=')) and self.mapping_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_global_localization_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger localizing the robot to a new pose that is near its current pose on the map
        if ((c == '-') or (c == '_')) and self.mapping_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_local_localization_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger driving the robot to the estimated next best place to perform a 3D scan
        if ((c == '\\') or (c == '|')) and self.mapping_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_drive_to_scan_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger performing a 3D scan using the D435i
        if (c == ' ') and self.mapping_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_head_scan_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger rotating the mobile base to align with the nearest 3D cliff detected visually
        if ((c == '[') or (c == '{')) and self.mapping_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_align_with_nearest_cliff_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # DEPRECATED: Trigger extend arm until contact
        if ((c == ']') or (c == '}')) and self.mapping_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_reach_until_contact_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # DEPRECATED: Trigger lower arm until contact
        if ((c == ':') or (c == ';')) and self.mapping_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_lower_until_contact_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'
            
        
        ####################################################
        ## OTHER CAPABILITIES
        ####################################################

        # Trigger Hello World whiteboard writing demo
        if ((c == '`') or (c == '~')) and self.hello_world_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_write_hello_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger open drawer demo with downward hook motion
        if ((c == 'z') or (c == 'Z')) and self.open_drawer_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_open_drawer_down_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger open drawer demo with upward hook motion
        if ((c == '.') or (c == '>')) and self.open_drawer_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_open_drawer_up_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger clean surface demo
        if ((c == '/') or (c == '?')) and self.clean_surface_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_clean_surface_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'
            
        # Trigger grasp object demo    
        if ((c == '\'') or (c == '\"')) and self.grasp_object_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_grasp_object_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'

        # Trigger deliver object demo    
        if ((c == 'y') or (c == 'Y')) and self.deliver_object_on:
            trigger_request = TriggerRequest() 
            trigger_result = node.trigger_deliver_object_service(trigger_request)
            rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            command = 'service_occurred'
       
        ####################################################
        ## BASIC KEYBOARD TELEOPERATION COMMANDS
        ####################################################
        
        # 8 or up arrow
        if c == 'i' or c == '\x1b[A':
            command = {'joint': 'joint_lift', 'delta': self.get_deltas()['translate']}
        # 2 or down arrow
        if c == 'k' or c == '\x1b[B':
            command = {'joint': 'joint_lift', 'delta': -self.get_deltas()['translate']}
        # if self.mode == 'manipulation':
        #     # 4 or left arrow
        #     if c == 'a' or c == '\x1b[D':
        #         command = {'joint': 'joint_mobile_base_translation', 'delta': self.get_deltas()['translate']}
        #     # 6 or right arrow
        #     if c == 'd' or c == '\x1b[C':
        #         command = {'joint': 'joint_mobile_base_translation', 'delta': -self.get_deltas()['translate']}
        if self.mode == 'position':
            # 4 or left arrow
            if c == 'a' or c == '\x1b[D':
                command = {'joint': 'translate_mobile_base', 'inc': self.get_deltas()['translate']}
            # 6 or right arrow
            if c == 'd' or c == '\x1b[C':
                command = {'joint': 'translate_mobile_base', 'inc': -self.get_deltas()['translate']}
            # 1 or end key 
            # if c == '7' or c == '\x1b[H':
            #     command = {'joint': 'rotate_mobile_base', 'inc': self.get_deltas()['rad']}
            # # 3 or pg down 5~
            # if c == '9' or c == '\x1b[5':
            #     command = {'joint': 'rotate_mobile_base', 'inc': -self.get_deltas()['rad']}
        # elif self.mode == 'navigation':
        #     rospy.loginfo('ERROR: Navigation mode is not currently supported.')

        if c == 'w' or c == 'W':
            command = {'joint': 'wrist_extension', 'delta': self.get_deltas()['translate']}
        if c == 's' or c == 'X':
            command = {'joint': 'wrist_extension', 'delta': -self.get_deltas()['translate']}
        # if c == 'd' or c == 'D':
        #     command = {'joint': 'joint_wrist_yaw', 'delta': -self.get_deltas()['rad']}
        # if c == 'a' or c == 'A':
        #     command = {'joint': 'joint_wrist_yaw', 'delta': self.get_deltas()['rad']}
        # if c == 'v' or c == 'V':
        #     command = {'joint': 'joint_wrist_pitch', 'delta': -self.get_deltas()['rad']}
        # if c == 'c' or c == 'C':
        #     command = {'joint': 'joint_wrist_pitch', 'delta': self.get_deltas()['rad']}
        if c == 'l' or c == 'P':
            command = {'joint': 'joint_wrist_roll', 'delta': self.get_deltas()['rad']}
        if c == 'j' or c == 'O':
            command = {'joint': 'joint_wrist_roll', 'delta': -self.get_deltas()['rad']}
        if c == 'm' or c == '\x1b[E' or c == 'g' or c == 'G':
            # grasp
            command = {'joint': 'joint_gripper_finger_left', 'delta': -self.get_deltas()['rad']}
        if c == 'n' or c == '\x1b[2' or c == 'r' or c == 'R':
            # release
            command = {'joint': 'joint_gripper_finger_left', 'delta': self.get_deltas()['rad']}
        # if c == 'i' or c == 'I':
        #     command = {'joint': 'joint_head_tilt', 'delta': (2.0 * self.get_deltas()['rad'])}
        # if c == ',' or c == '<':
        #     command = {'joint': 'joint_head_tilt', 'delta': -(2.0 * self.get_deltas()['rad'])}
        # if c == 'j' or c == 'J':
        #     command = {'joint': 'joint_head_pan', 'delta': (2.0 * self.get_deltas()['rad'])}
        # if c == 'l' or c == 'L':
        #     command = {'joint': 'joint_head_pan', 'delta': -(2.0 * self.get_deltas()['rad'])}
            
        # if c == 'b' or c == 'B':
        #     rospy.loginfo('process_keyboard.py: changing to BIG step size')
        #     self.step_size = 'big'
        # if c == 'm' or c == 'M':
        #     rospy.loginfo('process_keyboard.py: changing to MEDIUM step size')
        #     self.step_size = 'medium'
        # if c == 's' or c == 'S':
        #     rospy.loginfo('process_keyboard.py: changing to SMALL step size')
        #     self.step_size = 'small'
            
        if c == 'q' or c == 'Q':
            rospy.loginfo('keyboard_teleop exiting...')
            pub = rospy.Publisher('end_recording', String, queue_size=10)
            pub.publish('end')
            # for p in procs:
            #     p.kill()
            rospy.signal_shutdown('Received quit character (q), so exiting')
            self.file.close()

        ####################################################

        return command


class KeyboardTeleopNode(hm.HelloNode):

    def __init__(self, mapping_on=False, hello_world_on=False, open_drawer_on=False, clean_surface_on=False, grasp_object_on=False, deliver_object_on=False):
        hm.HelloNode.__init__(self)
        self.keys = GetKeyboardCommands(mapping_on, hello_world_on, open_drawer_on, clean_surface_on, grasp_object_on, deliver_object_on)
        self.rate = 10.0
        self.joint_state = None
        self.mapping_on = mapping_on
        self.hello_world_on = hello_world_on
        self.open_drawer_on = open_drawer_on
        self.clean_surface_on = clean_surface_on
        self.grasp_object_on = grasp_object_on
        self.deliver_object_on = deliver_object_on

    def joint_states_callback(self, joint_state):
        self.joint_state = joint_state

    def send_command(self, command):
        joint_state = self.joint_state
        if (joint_state is not None) and (command is not None) and (isinstance(command, dict)):
            point = JointTrajectoryPoint()
            point.time_from_start = rospy.Duration(0.0)
            trajectory_goal = FollowJointTrajectoryGoal()
            trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
            
            joint_name = command['joint']
            trajectory_goal.trajectory.joint_names = [joint_name]
            if 'inc' in command:
                inc = command['inc']
                new_value = inc
            elif 'delta' in command:
                joint_index = joint_state.name.index(joint_name)
                joint_value = joint_state.position[joint_index]
                delta = command['delta']
                new_value = joint_value + delta
            point.positions = [new_value]
            trajectory_goal.trajectory.points = [point]
            trajectory_goal.trajectory.header.stamp = rospy.Time.now()
            self.trajectory_client.send_goal(trajectory_goal)

    def main(self):
        hm.HelloNode.main(self, 'keyboard_teleop', 'keyboard_teleop', wait_for_first_pointcloud=False)
        import time
        

        if self.mapping_on: 
            rospy.loginfo('Node ' + self.node_name + ' waiting to connect to /funmap/trigger_head_scan.')

            rospy.wait_for_service('/funmap/trigger_head_scan')
            rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_head_scan.')
            self.trigger_head_scan_service = rospy.ServiceProxy('/funmap/trigger_head_scan', Trigger)

            rospy.wait_for_service('/funmap/trigger_drive_to_scan')
            rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_drive_to_scan.')
            self.trigger_drive_to_scan_service = rospy.ServiceProxy('/funmap/trigger_drive_to_scan', Trigger)

            rospy.wait_for_service('/funmap/trigger_global_localization')
            rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_global_localization.')
            self.trigger_global_localization_service = rospy.ServiceProxy('/funmap/trigger_global_localization', Trigger)

            rospy.wait_for_service('/funmap/trigger_local_localization')
            rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_local_localization.')
            self.trigger_local_localization_service = rospy.ServiceProxy('/funmap/trigger_local_localization', Trigger)

            rospy.wait_for_service('/funmap/trigger_align_with_nearest_cliff')
            rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_align_with_nearest_cliff.')
            self.trigger_align_with_nearest_cliff_service = rospy.ServiceProxy('/funmap/trigger_align_with_nearest_cliff', Trigger)

            rospy.wait_for_service('/funmap/trigger_reach_until_contact')
            rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_reach_until_contact.')
            self.trigger_reach_until_contact_service = rospy.ServiceProxy('/funmap/trigger_reach_until_contact', Trigger)

            rospy.wait_for_service('/funmap/trigger_lower_until_contact')
            rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_lower_until_contact.')
            self.trigger_lower_until_contact_service = rospy.ServiceProxy('/funmap/trigger_lower_until_contact', Trigger)

        if self.hello_world_on: 
            rospy.wait_for_service('/hello_world/trigger_write_hello')
            rospy.loginfo('Node ' + self.node_name + ' connected to /hello_world/trigger_write_hello.')
            self.trigger_write_hello_service = rospy.ServiceProxy('/hello_world/trigger_write_hello', Trigger)

        if self.open_drawer_on:
            rospy.wait_for_service('/open_drawer/trigger_open_drawer_down')
            rospy.loginfo('Node ' + self.node_name + ' connected to /open_drawer/trigger_open_drawer_down.')
            self.trigger_open_drawer_down_service = rospy.ServiceProxy('/open_drawer/trigger_open_drawer_down', Trigger)

            rospy.wait_for_service('/open_drawer/trigger_open_drawer_up')
            rospy.loginfo('Node ' + self.node_name + ' connected to /open_drawer/trigger_open_drawer_up.')
            self.trigger_open_drawer_up_service = rospy.ServiceProxy('/open_drawer/trigger_open_drawer_up', Trigger)

            
        if self.clean_surface_on:
            rospy.wait_for_service('/clean_surface/trigger_clean_surface')
            rospy.loginfo('Node ' + self.node_name + ' connected to /clean_surface/trigger_clean_surface.')
            self.trigger_clean_surface_service = rospy.ServiceProxy('/clean_surface/trigger_clean_surface', Trigger)

        if self.grasp_object_on:
            rospy.wait_for_service('/grasp_object/trigger_grasp_object')
            rospy.loginfo('Node ' + self.node_name + ' connected to /grasp_object/trigger_grasp_object.')
            self.trigger_grasp_object_service = rospy.ServiceProxy('/grasp_object/trigger_grasp_object', Trigger)

        if self.deliver_object_on:
            rospy.wait_for_service('/deliver_object/trigger_deliver_object')
            rospy.loginfo('Node ' + self.node_name + ' connected to /deliver_object/trigger_deliver_object.')
            self.trigger_deliver_object_service = rospy.ServiceProxy('/deliver_object/trigger_deliver_object', Trigger)

        rospy.Subscriber('/stretch/joint_states', JointState, self.joint_states_callback)


        rate = rospy.Rate(self.rate)
        
        pub = rospy.Publisher('end_recording', String, queue_size=10)
        for i in range(2):
            pub.publish(f'start.{run_id}')
            time.sleep(1)
        command = 1 # set equal to not None, so menu is printed out on first loop
        while not rospy.is_shutdown():
            if self.joint_state is not None:
                self.keys.print_commands(self.joint_state, command)
                command = self.keys.get_command(self)
                self.send_command(command)
            rate.sleep()


if __name__ == '__main__':

    from subprocess import Popen
    import sys
    import os

    import rospy
    
    run_id = sys.argv[1]
    # try:
    #     os.rmdir(f'data/{run_id}/')
    # except:
    #     print("Directory does not exist")
    os.makedirs(Data_PATH + f'/{run_id}/video', exist_ok=True)
    
    # commands = [f'python3 grab_audio.py {run_id}']
    # procs = [ Popen(i, shell = True) for i in commands ]
    
    try:
        parser = ap.ArgumentParser(description='Keyboard teleoperation for stretch.')
        parser.add_argument('--mapping_on', action='store_true', help='Turn on mapping control. For example, the space bar will trigger a head scan. This requires that the mapping node be run (funmap).')
        parser.add_argument('--hello_world_on', action='store_true', help='Enable Hello World writing trigger, which requires connection to the appropriate hello_world service.')
        parser.add_argument('--open_drawer_on', action='store_true', help='Enable Open Drawer trigger, which requires connection to the appropriate open_drawer service.')
        parser.add_argument('--clean_surface_on', action='store_true', help='Enable Clean Surface trigger, which requires connection to the appropriate clean_surface service.')
        parser.add_argument('--grasp_object_on', action='store_true', help='Enable Grasp Object trigger, which requires connection to the appropriate grasp_object service.')
        parser.add_argument('--deliver_object_on', action='store_true', help='Enable Deliver Object trigger, which requires connection to the appropriate deliver_object service.')

        args, unknown = parser.parse_known_args()
        mapping_on = args.mapping_on
        hello_world_on = args.hello_world_on
        open_drawer_on = args.open_drawer_on
        clean_surface_on = args.clean_surface_on
        grasp_object_on = args.grasp_object_on
        deliver_object_on = args.deliver_object_on

        node = KeyboardTeleopNode(mapping_on, hello_world_on, open_drawer_on, clean_surface_on, grasp_object_on, deliver_object_on)
        node.main()
        
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')