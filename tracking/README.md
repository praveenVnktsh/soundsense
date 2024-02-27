### Run ORB-SLAM3

Start the container

```bash
sudo xhost +local:root && \
docker run --privileged --name orb-3-container \
--rm -p 8087:8087 \
-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev:/dev:ro \
-v /home/praveen/dev/mmml/soundsense/data/:/dpds/data \
--gpus all \
-it lmwafer/orb-slam-3-ready:1.0-ubuntu18.04
```

Run the code:
<!-- path_to_sequence -->
```bash
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt /dpds/data/orbslam3/dji.yaml /dpds/data/run1/ 
```