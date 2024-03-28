

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
 
Terminal 1 - roslaunch stretch_core stretch_driver.launch
terminal 2 - grab_audio.py
terminal 3 - teleop_collector.py (control on this)
