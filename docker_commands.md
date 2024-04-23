### Instructions
Go to ~/Soundsense/soundsense

Terminal 1 
- docker run -it --mount type=bind,source="$(pwd)",target=/home/soundsense ros:noetic-robot
- roscore

Terminal 2 

Identify the name of the docker container in the NAMES column
- docker ps -l
- docker exec -it <name> bash
- source /opt/ros/noetic/setup.bash

