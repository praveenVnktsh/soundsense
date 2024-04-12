

Directory structure:

```bash
models
    baselines
        cnnlstm
            model.py
            ...
        mulsa
            model.py
            ...
test
    test_*.py
model_inference.py        
```




#### Keyboard teleop

w - move up
s - move down
a - move left
d - move right

i - extend arm
j - retract arm

l - roll right
j - roll left

m - close gripper
n - open gripper


### Instructions
You will need 3 terminals:
Terminal 1 - roslaunch stretch_core stretch_driver.launch (background terminal)

Split terminal into two panels:
terminal 2 - grab_audio.py

terminal 3 - teleop_collector.py (control on this)

After collecting one data point, ensure that the audio recorded is correct (play music near the camera and see if it is correct). Once everything looks okay, you can start collecting all the datapoints.



### README

```shell
roslaunch audio_capture capture.launch format:="wave"
```
