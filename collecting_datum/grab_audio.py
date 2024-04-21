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

rospy.init_node('grab_audio', anonymous=True)
def callBackEndRecording(data: String):
	# if comething was publish in this topic, stop recording
	global reccord, w, path, inp
	print("received", data.data)

	if data.data.split('.')[0] == 'start':
		print("Started recording")
		inp = alsaaudio.PCM(1)
		inp.setchannels(1)
		inp.setrate(48000)
		inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
		inp.setperiodsize(1024)
		
		run_id = data.data.split('.')[1]
		pathFolder = '../data/sorting/' + run_id + '/'
		os.makedirs(pathFolder, exist_ok= True)
		os.makedirs(pathFolder + 'video/', exist_ok= True)
		path = pathFolder + f"{datetime.datetime.now().strftime('%s%f')}.wav"
		w = wave.open(path, 'w')
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(48000)
		reccord = True
	else:
		inp.close()
		w.close()
		w = None
		reccord = False
	


	
sub = rospy.Subscriber('end_recording', String, callBackEndRecording)


reccord = False

if __name__ == '__main__':


	# date = datetime.datetime.now().strftime("%s")
	# while date[-1] != '0':
	# 	date = datetime.datetime.now().strftime("%s")
	# print("Starting audio capture at", date)
	inp = None
	
	# create listener
	w = None
	
	while not rospy.is_shutdown():
		
			
		while w is not None and reccord == True:
			try:
				l, data = inp.read()
			# a = numpy.fromstring(data, dtype='int16')
				w.writeframes(data)
			except:
				continue


	
		
			

