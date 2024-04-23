### Instructions
Go to ~/Soundsense/soundsense on server.

Terminal 1 
- docker build -t mmml:ros-noetic .
- docker run -it --network="host" --mount type=bind,source="$(pwd)",target=/home/soundsense mmml:ros-noetic
- roscore

Terminal 2 

Identify the name of the docker container in the NAMES column
- docker ps -l
- docker exec -it <name> bash
- source /opt/ros/noetic/setup.bash

To check if server and robot are communicating
- rostopic pub -r 10 my_topic std_msgs/String "hello there"

Run the inference code
- cd /home/soundsense/
- python3 models/baselines/mulsa/inference_ros.py


### Robot terminal
- export ROS_MASTER_URI=http://172.26.32.75:11311/