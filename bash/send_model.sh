#!/bin/bash  
MODEL="/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/lightning_logs/"
MODEL+="sorting_imi_vg_ag_lstm_seqlen_3_mha_spec_pretrained04-24-17:22:04"
rsync -r $MODEL  hello-robot@172.26.163.219:/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models/