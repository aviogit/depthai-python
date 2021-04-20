#!/usr/bin/env python3

import cv2
import sys
import time
from time import sleep

from datetime import datetime

from random import randrange

import argparse
from argument_parser import var2opt, define_boolean_argument

from colormaps import apply_colormap

parser = argparse.ArgumentParser(description='Extract a color/disparity pair from two (hopefully synchronized) color/depth video files.')
parser.add_argument('filenames',		nargs=2)
parser.add_argument('--start-frame',		default=0, type=int, help='start frame for the random interval (default: 0)')
parser.add_argument('--end-frame',		default=0, type=int, help='end   frame for the random interval (default: last frame)')

define_boolean_argument(parser, *var2opt('sequential'), 'show frames sequentially from --start-frame to --end-frame', False)
define_boolean_argument(parser, *var2opt('continuous'), 'show frames continuously from --start-frame to --end-frame', False)

args = parser.parse_args()

color_video_fname = args.filenames[0]
depth_video_fname = args.filenames[1]

color_cap = depth_cap = color_frame = depth_frame = None

def get_num_frames(video_fname):
	print(video_fname)
	cap = cv2.VideoCapture(video_fname)
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	return cap, num_frames

def get_frame_no(cap, frame_number):
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

pause = False

while True:
	delay_ms = 1 if args.continuous else 0
	if pause:
		delay_ms = 100
	key = cv2.waitKey(delay_ms)
	if key & 0xFF == ord('q'):
		break
	if key & 0xFF == ord('p'):
		pause = not pause
	if pause:
		continue

	if not args.sequential:
		random_frame_no   = randrange(start_frame, end_frame)
		print(f'{color_num_frames = } - {random_frame_no = }')

		cres, color_frame = get_frame_no(color_cap, random_frame_no)
		print(f'{cres = } - {color_frame.shape = }')
	
		dres, depth_frame = get_frame_no(depth_cap, random_frame_no)
		print(f'{cres = } - {depth_frame.shape = }')
	else:
		if color_frame is None and depth_frame is None:
			cres, color_frame = get_frame_no(color_cap, start_frame)
			dres, depth_frame = get_frame_no(depth_cap, start_frame)
		else:
			cres, color_frame = color_cap.read()
			dres, depth_frame = depth_cap.read()
			print(f'Showing frame: {color_cap.get(cv2.CAP_PROP_POS_FRAMES)}')

	depth_frame = apply_colormap(depth_frame, cmap=0)	# 13 is cool but it's too dark
	
	color_width, color_height, color_ch = color_frame.shape
	depth_width, depth_height, depth_ch = depth_frame.shape
	cv2.namedWindow ('rgb',		cv2.WINDOW_NORMAL)
	cv2.resizeWindow('rgb',		int(color_width/2), int(color_height/2))
	cv2.namedWindow ('disparity',	cv2.WINDOW_NORMAL)
	cv2.resizeWindow('disparity',	int(color_width/2), int(color_height/2))

	cv2.imshow('rgb',	color_frame)
	cv2.imshow('disparity',	depth_frame)

