#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np
from pathlib import Path

from argument_parser import define_boolean_argument, var2opt

from utils import wlsFilter, apply_wls_filter

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

	cv2.imshow('frame', combo)

depth_cap.release()
rright_cap.release()

cv2.destroyAllWindows()

sys.exit(0)


