#!/usr/bin/env python

import sys
import cv2
import glob
import argparse
import datetime
import numpy as np
from pathlib import Path

from argument_parser import define_boolean_argument, var2opt


# Launch with:

# /mnt/btrfs-data/venvs/ml-tutorials/repos/depthai-python/examples/custom-scripts/left-right-color-video-parser.py --prefix 2021-06-08-17-16-44 --rectright --no-disparity --no-wls-disparity --rect --mp4 --start-frame 2800


def get_quarter_img(frame, show_quarter_img):
	if show_quarter_img:
		frame_s = frame[int(frame.shape[0]/4):frame.shape[1], :]
	else:
		frame_s = frame
	return frame_s



parser = argparse.ArgumentParser()
parser.add_argument('--prefix', nargs='?', help="color-/left-/right-<prefix>.h265(.mp4) video files will be used")
parser.add_argument('--start-frame', default=0, type=int, help='start frame for start replaying the video triplet')
parser.add_argument('--resize', default='', type=str, help='resize image pairs to WxH resolution (e.g. --resize 640x480')
define_boolean_argument(parser, *var2opt('disparity'), 'capture disparity instead of left/right streams', False)
define_boolean_argument(parser, *var2opt('wls_disparity'), 'capture wls disparity instead of left/right or normal disparity streams', True)
define_boolean_argument(parser, *var2opt('rectright'), 'capture rectright instead of color stream', True)
define_boolean_argument(parser, *var2opt('rect'), 'prepend the "rect" prefix to left and right to open rectleft and rectright instead of just left/right', False)
define_boolean_argument(parser, *var2opt('mp4'), 'postpone the "mp4" suffix to file names to open .h265.mp4 files instead of just .h265 files', False)
args = parser.parse_args()

if args.prefix is None:
	print(f'Please specify a valid prefix for video files. Exiting...')
	sys.exit(0)

if args.mp4:
	mp4_suffix	= '.mp4'
else:
	mp4_suffix	= ''

if args.rectright:
	main_fn	= glob.glob(f'rectright-{args.prefix}*h265{mp4_suffix}')
else:
	main_fn	= glob.glob(f'color-{args.prefix}*h265{mp4_suffix}')

print(f'Found main file(s): {main_fn}')
main_fn = main_fn[0]
print(f'Opening main file: {main_fn}')
cap			= cv2.VideoCapture(f'{main_fn}')
caplen			= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if args.disparity:
	depth_cap	= cv2.VideoCapture(f'depth-{args.prefix}.h265{mp4_suffix}')
	depth_len	= int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f'{caplen = } - {depth_len = }')
elif args.wls_disparity:
	wls_fn		= glob.glob(f'wls-{args.prefix}*')
	print(f'Found disparity file(s): {wls_fn}')
	wls_fn		= wls_fn[0]
	print(f'Opening disparity file: {wls_fn}')
	wls_cap		= cv2.VideoCapture(f'{wls_fn}')
	wls_len		= int(wls_cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f'{caplen = } - {wls_len = }')
else:
	if args.rect:
		rect_prefix = 'rect'
	else:
		rect_prefix = ''
	left_fn		= f'{rect_prefix}left-{args.prefix}.h265{mp4_suffix}'
	right_fn	= f'{rect_prefix}right-{args.prefix}.h265{mp4_suffix}'
	#left_fn		= f'left-{args.prefix}.h265'
	#right_fn	= f'right-{args.prefix}.h265'
	print(f'Opening left/right files: {left_fn}/{right_fn}')
	left_cap	= cv2.VideoCapture(f'{left_fn}')
	left_len	= int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
	right_cap	= cv2.VideoCapture(f'{right_fn}')
	right_len	= int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f'{caplen = } - {left_len = } - {right_len = }')

show_quarter_img = False

pause = False

if args.start_frame != 0:
	print(f'Start frame: {args.start_frame}')
	cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
	if args.disparity:
		depth_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
	elif args.wls_disparity:
		wls_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
	else:
		left_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)
		right_cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame-1)

frame_counter = args.start_frame

while cap.isOpened():
	if args.disparity:
		if not depth_cap.isOpened():
			break
	elif args.wls_disparity:
		if not wls_cap.isOpened():
			break
	else:
		if not left_cap.isOpened() or not right_cap.isOpened():
			break

	delay_ms = 1 #if args.continuous else 0
	if pause:
		delay_ms = 100
	key = cv2.waitKey(delay_ms)
	if key & 0xFF == ord('q'):
		break
	if key & 0xFF == ord('p'):
		pause = not pause
		print(f'Frame no.: {frame_counter}')
	if pause:
		continue

	cret, cframe   = cap.read()
	frame_counter += 1
	if args.disparity:
		dret, dframe = depth_cap.read()
		small_size = (dframe.shape[1], dframe.shape[0])
	elif args.wls_disparity:
		wret, wframe = wls_cap.read()
		small_size = (wframe.shape[1], wframe.shape[0])
	else:
		lret, lframe = left_cap.read()
		rret, rframe = right_cap.read()
		small_size = (cframe.shape[1], cframe.shape[0])

	cframe   = cv2.resize(cframe, small_size)
	cframe_s = get_quarter_img(cframe, show_quarter_img)

	if args.disparity:
		dframe_s = get_quarter_img(dframe, show_quarter_img)
		dframe_s = cv2.medianBlur(dframe_s, 5)
		dframe_s = cv2.normalize(dframe_s, None, 0, 255, cv2.NORM_MINMAX)
		dframe_s = cv2.applyColorMap(dframe_s, cv2.COLORMAP_JET)

		combo = np.concatenate((cframe_s, dframe_s), axis=0)
	elif args.wls_disparity:
		wframe_s = get_quarter_img(wframe, show_quarter_img)
		combo = np.concatenate((cframe_s, wframe_s), axis=0)
	else:
		lframe_s = get_quarter_img(lframe, show_quarter_img)
		rframe_s = get_quarter_img(rframe, show_quarter_img)
		combo = np.concatenate((lframe_s, cframe_s), axis=0)
		combo = np.concatenate((combo,    rframe_s), axis=0)

	if args.resize != '':
		new_size = args.resize.split('x')
		new_size = [int(x) for x in new_size]
		combo = cv2.resize(combo, tuple(new_size))

	if key & 0xFF == ord('s'):
		cv2.imwrite(f'/tmp/l-{frame_counter}.jpg', lframe);
		cv2.imwrite(f'/tmp/r-{frame_counter}.jpg', rframe);
		print(f'Saved frame no.: {frame_counter}')

	cv2.imshow('frame', combo)
	if frame_counter % 1000 == 0:
		print(f'Frame no.: {frame_counter}')

cap.release()
if args.disparity:
	depth_cap.release()
elif args.wls_disparity:
	wls_cap.release()
else:
	left_cap.release()
	right_cap.release()

cv2.destroyAllWindows()

sys.exit(0)


