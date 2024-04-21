#!/usr/bin/env python
#coding: utf-8

'''
author: thibault asselborn
date: 6.10.2016
description: Node used to record sound from a device using a rospy node. Work with alsaaudio lib.

'''

import os, rospy, datetime, alsaaudio, wave, numpy
from os.path import expanduser
from std_msgs.msg import String
import cv2
import sys
import rospy
import numpy as np
from audio_common_msgs.msg import AudioData

rospy.init_node('grab_audio', anonymous=True)
def callBackEndRecording(data: String):
	# if comething was publish in this topic, stop recording
	global reccord, w, path, buffer
	print("received", data.data)

	if data.data.split('.')[0] == 'start':
		print("Started recording")
		run_id = data.data.split('.')[1]
		pathFolder = '../data/sorting/' + run_id + '/'
		os.makedirs(pathFolder, exist_ok= True)
		os.makedirs(pathFolder + 'video/', exist_ok= True)
		path = pathFolder + f"{datetime.datetime.now().strftime('%s%f')}.wav"
		
		reccord = True
		buffer = []
	else:
		reccord = False

		with wave.open(path, 'wb') as wf:
			wf.setnchannels(1)  
			wf.setsampwidth(2)  
			wf.setframerate(16000)
			origbuf = np.array(buffer.copy(), dtype = np.int16)
			wf.writeframes(origbuf.tobytes())

buffer = []
def callback( data):
	global reccord, buffer
	if reccord:
		# audio = np.frombuffer(data.data, dtype=np.int16).copy().astype(np.float64)
		# audio /= 32768
		# audio = audio.tolist()
		audio = np.frombuffer(data.data, dtype=np.int16).tolist()
		buffer += (audio.copy())

audio_sub = rospy.Subscriber('/audio/audio', AudioData, callback)

sub = rospy.Subscriber('end_recording', String, callBackEndRecording)

reccord = False
path = None

if __name__ == '__main__':


	# create listener
	w = None
	print("Waiting to start record")
	try:	
		rospy.spin()
	except KeyboardInterrupt:
		if w is not None:
			w.close()
			w = None
		sys.exit(0)
	
		
			

