#!/bin/bash  
MODEL="/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/lightning_logs/"
MODEL+="c"
rsync -r $MODEL  hello-robot@172.26.163.219:/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models/