#!/usr/bin/env python3

import cv2
import sys
import time

from datetime import datetime

from random import randrange

import argparse

from colormaps import apply_colormap

parser = argparse.ArgumentParser(description='Extract a color/disparity pair from two (hopefully synchronized) color/depth video files.')
parser.add_argument('filenames',		nargs=2)
#parser.add_argument('--color-video-filename',	nargs='?', default='color.h265')
#parser.add_argument('--depth-video-filename',	nargs='?', default='depth.h265')
parser.add_argument('--start-frame',		default=0, type=int, help='start frame for the random interval')
parser.add_argument('--end-frame',		default=0, type=int, help='end frame for the random interval')

args = parser.parse_args()

color_video_fname = args.filenames[0]
depth_video_fname = args.filenames[1]
#color_video_fname = 'color-2021-04-08-16-26-06.h265-60fps.mp4'
#depth_video_fname = 'depth-2021-04-08-16-26-06.h265-120fps.mp4'
#color_video_fname = 'color-2021-04-08-16-26-06.h265'
#depth_video_fname = 'depth-2021-04-08-16-26-06.h265'

color_cap = depth_cap = None

def get_num_frames(video_fname):
	print(video_fname)
	cap = cv2.VideoCapture(video_fname)
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	return cap, num_frames

def get_frame_no(cap, frame_number):
	#cap = cv2.VideoCapture(video_fname)
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
	res, frame = cap.read()
	return res, frame

color_cap, color_num_frames = get_num_frames(color_video_fname)
depth_cap, depth_num_frames = get_num_frames(depth_video_fname)
print(f'{color_num_frames = } - {depth_num_frames = }')
if color_num_frames != depth_num_frames:
	print(f'-------------------------------------------------------------------')
	print(f'Warning! The number of frames in color and depth video file differ!')
	print(f'-------------------------------------------------------------------')
if color_num_frames <= 0:
	print(f'Error retrieving the number of frames in color video file: {color_video_fname}')
	sys.exit(0)

start_frame = args.start_frame if args.start_frame else 0
end_frame   = args.end_frame if args.end_frame else color_num_frames

while True:
	random_frame_no   = randrange(start_frame, end_frame)
	print(f'{color_num_frames = } - {random_frame_no = }')

	cres, color_frame = get_frame_no(color_cap, random_frame_no)
	print(f'{cres = } - {color_frame.shape = }')
	
	cres, depth_frame = get_frame_no(depth_cap, random_frame_no)
	print(f'{cres = } - {depth_frame.shape = }')

	depth_frame = apply_colormap(depth_frame, cmap=0)	# 13 is cool but it's too dark
	
	color_width, color_height, color_ch = color_frame.shape
	depth_width, depth_height, depth_ch = depth_frame.shape
	cv2.namedWindow ('rgb',		cv2.WINDOW_NORMAL)
	cv2.resizeWindow('rgb',		int(color_width/2), int(color_height/2))
	cv2.namedWindow ('disparity',	cv2.WINDOW_NORMAL)
	cv2.resizeWindow('disparity',	int(color_width/2), int(color_height/2))

	#cv2.imshow(video_fname+' frame '+ str(frame_seq), frame)
	cv2.imshow('rgb',	color_frame)
	cv2.imshow('disparity',	depth_frame)

	key = cv2.waitKey(0)
	if key & 0xFF == ord('q'):
		break
