#!/usr/bin/env python3

import cv2
import sys
import time

from datetime import datetime

from random import randrange


color_video_fname = 'color-2021-04-08-16-26-06.h265-60fps.mp4'
depth_video_fname = 'depth-2021-04-08-16-26-06.h265-120fps.mp4'
#color_video_fname = 'color-2021-04-08-16-26-06.h265'
#depth_video_fname = 'depth-2021-04-08-16-26-06.h265'

def get_num_frames(video_fname):
	cap = cv2.VideoCapture(video_fname)
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	return num_frames

def get_frame_no(video_fname, frame_number):
	cap = cv2.VideoCapture(video_fname)
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
	res, frame = cap.read()
	return res, frame

color_num_frames  = get_num_frames(color_video_fname)
print(f'{color_num_frames = }')
random_frame_no   = randrange(color_num_frames)
print(f'{color_num_frames = } - {random_frame_no = }')
cres, color_frame = get_frame_no(color_video_fname, random_frame_no)
print(f'{cres = } - {color_frame.shape = }')

depth_num_frames  = get_num_frames(depth_video_fname)
print(f'{depth_num_frames = } - {random_frame_no = }')
cres, depth_frame = get_frame_no(depth_video_fname, random_frame_no)
print(f'{cres = } - {depth_frame.shape = }')

color_width, color_height, color_ch = color_frame.shape
depth_width, depth_height, depth_ch = depth_frame.shape

cv2.namedWindow('rgb',cv2.WINDOW_NORMAL)
cv2.resizeWindow('rgb', int(color_width/2), int(color_height/2))

cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disparity', int(color_width/2), int(color_height/2))

while cv2.waitKey(0) & 0xFF != ord('q'):
	#cv2.imshow(video_fname+' frame '+ str(frame_seq), frame)
	cv2.imshow('rgb',	color_frame)
	cv2.imshow('disparity',	depth_frame)

