#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np
from pathlib import Path

from argument_parser import define_boolean_argument, var2opt

from utils import wlsFilter, apply_wls_filter

def get_quarter_img(frame, show_quarter_img):
	if show_quarter_img:
		frame_s = frame[int(frame.shape[0]/4):frame.shape[1], :]
	else:
		frame_s = frame
	return frame_s


parser = argparse.ArgumentParser()
parser.add_argument('--prefix', nargs='?', help="color-/left-/right-<prefix>.h265 video files will be used")
parser.add_argument('--start-frame', default=0, type=int, help='start frame for start replaying the video triplet')
define_boolean_argument(parser, *var2opt('disparity'), 'capture disparity instead of left/right streams', False)
define_boolean_argument(parser, *var2opt('wls_disparity'), 'capture wls disparity instead of left/right or normal disparity streams', True)
define_boolean_argument(parser, *var2opt('show_wls_preview'), 'show host-side WLS filtering made with OpenCV', False)
args = parser.parse_args()

if args.prefix is None:
	print(f'Please specify a valid prefix for video files. Exiting...')
	sys.exit(0)



wlsFilter = wlsFilter(args, _lambda=8000, _sigma=1.5)

baseline = 75 #mm
disp_levels = 96
fov = 71.86



depth_cap  = cv2.VideoCapture(f'depth-{args.prefix}.h265')
depth_len  = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
rright_cap = cv2.VideoCapture(f'rectright-{args.prefix}.h265')
rright_len = int(rright_cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'{rright_len = } - {depth_len = }')

#small_size = (1280, 720)
small_size = (640, 400)

pause = False

'''
if args.start_frame != 0:
	print(f'Start frame: {args.start_frame}')
	color_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
	if args.disparity:
		depth_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
	elif args.wls_disparity:
		wls_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
	else:
		left_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
		right_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
'''

while depth_cap.isOpened() and rright_cap.isOpened():

	delay_ms = 1 #if args.continuous else 0
	if pause:
		delay_ms = 100
	key = cv2.waitKey(delay_ms)
	if key & 0xFF == ord('q'):
		break
	if key & 0xFF == ord('p'):
		pause = not pause
	if pause:
		continue

	dret,  dframe	= depth_cap.read()
	rrret, rrframe	= rright_cap.read()
	small_size	= (dframe.shape[1], dframe.shape[0])

	print(f'{dframe.shape} - {rrframe.shape} - {small_size}')

	disp_gray	= cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
	rr_img		= cv2.flip(rrframe, flipCode=1)
	filtered_disp, colored_disp = apply_wls_filter(wlsFilter, disp_gray, rr_img, baseline, fov, disp_levels, args)
	combo		= np.concatenate((rr_img, colored_disp), axis=0)

	'''
	cframe   = cv2.resize(cframe, small_size)
	#print(f'{small_size[1]/4} - {lframe.shape[1]}')
	#cframe_s = cframe[int(small_size[0]/4):cframe.shape[1], :]
	cframe_s = get_quarter_img(cframe, show_quarter_img)

	if args.disparity:
		#dframe_s = dframe[int(small_size[0]/4):dframe.shape[1], :]
		dframe_s = get_quarter_img(dframe, show_quarter_img)
		dframe_s = cv2.medianBlur(dframe_s, 5)
		dframe_s = cv2.normalize(dframe_s, None, 0, 255, cv2.NORM_MINMAX)
		dframe_s = cv2.applyColorMap(dframe_s, cv2.COLORMAP_JET)

	elif args.wls_disparity:
		wframe_s = get_quarter_img(wframe, show_quarter_img)
		combo = np.concatenate((cframe_s, wframe_s), axis=0)
	else:
		#lframe_s = lframe[int(small_size[0]/4):lframe.shape[1], :]
		#rframe_s = rframe[int(small_size[0]/4):rframe.shape[1], :]
		lframe_s = get_quarter_img(lframe, show_quarter_img)
		rframe_s = get_quarter_img(rframe, show_quarter_img)
		combo = np.concatenate((lframe_s, cframe_s), axis=0)
		combo = np.concatenate((combo,    rframe_s), axis=0)
	'''

	#print(f'{cframe_s.shape} - {lframe_s.shape} - {rframe_s.shape}')

	cv2.imshow('frame', combo)

depth_cap.release()
rright_cap.release()

cv2.destroyAllWindows()

sys.exit(0)


