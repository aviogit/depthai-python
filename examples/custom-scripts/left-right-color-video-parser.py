#!/usr/bin/env python

import sys
import cv2
import glob
import argparse
import datetime
import numpy as np
from pathlib import Path

from argument_parser import define_boolean_argument, var2opt

from colormaps import apply_colormap
from globber import globber

from optical_flow import optical_flow

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
parser.add_argument('--start-frame', default=0, type=int, help='start frame for start replaying the video pair (or triplet)')
parser.add_argument('--fps', default=30, type=int, help='use that FPS replaying the video pair (or triplet)')
parser.add_argument('--resize', default='', type=str, help='resize image pairs to WxH resolution (e.g. --resize 640x480')
define_boolean_argument(parser, *var2opt('disparity'), 'capture disparity instead of left/right streams', False)
define_boolean_argument(parser, *var2opt('wls_disparity'), 'capture wls disparity instead of left/right or normal disparity streams', True)
define_boolean_argument(parser, *var2opt('rectright'), 'capture rectright instead of color stream', True)
define_boolean_argument(parser, *var2opt('rect'), 'prepend the "rect" prefix to left and right to open rectleft and rectright instead of just left/right', False)
define_boolean_argument(parser, *var2opt('mp4'), 'postpone the "mp4" suffix to file names to open .h265.mp4 files instead of just .h265 files', False)
define_boolean_argument(parser, *var2opt('debug_optical_flow'), 'print what OpenCV optical flow is doing', False)
args = parser.parse_args()

print(f'Received arguments: {args}')

if args.prefix is None:
	print(f'Please specify a valid prefix for video files. Exiting...')
	sys.exit(0)

if args.mp4:
	mp4_suffix	= '.mp4'
else:
	mp4_suffix	= ''

'''
depth_fn = glob.glob(f'depth-{args.prefix}*h265{mp4_suffix}')
if args.rectright:
	main_fn	= glob.glob(f'rectright-{args.prefix}*h265{mp4_suffix}')
	if len(main_fn) == 0:
		main_fn  = glob.glob(f'*{args.prefix}-rectright*h265{mp4_suffix}')
		depth_fn = glob.glob(f'*{args.prefix}-disp*h265{mp4_suffix}')
else:
	main_fn	= glob.glob(f'color-{args.prefix}*h265{mp4_suffix}')
	if len(main_fn) == 0:
		main_fn	 = glob.glob(f'*{args.prefix}-color*h265{mp4_suffix}')
		depth_fn = glob.glob(f'*{args.prefix}-disp*h265{mp4_suffix}')
'''
main_fn, depth_fn = globber(args.prefix, args.rectright, mp4_suffix)

print(f'Found file(s): {main_fn} {depth_fn}')
main_fn = main_fn[0]
print(f'Opening main file: {main_fn}')

if args.debug_optical_flow and False:
	main_fn = '/mnt/btrfs-data/venvs/ml-tutorials/repos/depthai-python/examples/custom-scripts/optical-flow-example/slow_traffic_small.mp4'

cap			= cv2.VideoCapture(f'{main_fn}')
caplen			= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if args.disparity:
	depth_cap	= cv2.VideoCapture(f'{depth_fn[0]}')
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

if args.disparity or True:	# TODO: replace with args.optflow
	optflow, optflow_img = None, None

#print(f'Are main and disparity file opened? {cap.isOpened()} {depth_cap.isOpened()}')
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

	#delay_ms = 1 #if args.continuous else 0
	delay_ms = int(1000 / args.fps)
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

	cframe_s = cv2.resize(cframe, small_size)
	cframe_s = get_quarter_img(cframe_s, show_quarter_img)

	if args.disparity or True:	# TODO: replace with args.optflow
		if frame_counter >= 0:
			if optflow is None:
				optflow = optical_flow(cframe, debug=args.debug_optical_flow)
			else:
				optflow_img = optflow.do_opt_flow(cframe)
		if optflow_img is not None:
			cv2.imshow('optflow', optflow_img)

	if args.disparity:
		dframe_s = get_quarter_img(dframe, show_quarter_img)
		dframe_s = cv2.medianBlur(dframe_s, 5)
		dframe_s = cv2.normalize(dframe_s, None, 0, 255, cv2.NORM_MINMAX)
		#dframe_s = cv2.applyColorMap(dframe_s, cv2.COLORMAP_JET)
		dframe_s = apply_colormap(dframe_s, cmap=13)

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
		if args.disparity:
			dframe_colored = cv2.normalize(dframe, None, 0, 255, cv2.NORM_MINMAX)
			dframe_colored = apply_colormap(dframe_colored, cmap=13)
			cv2.imwrite(f'/tmp/d-{frame_counter}.jpg', dframe_colored);
		else:
			cv2.imwrite(f'/tmp/l-{frame_counter}.jpg', lframe);
		if args.rectright:
			cv2.imwrite(f'/tmp/r-{frame_counter}.jpg', rframe);
		else:
			cv2.imwrite(f'/tmp/c-{frame_counter}.jpg', cframe);
		print(f'Saved frame no.: {frame_counter}')

	cv2.putText(combo, f'{frame_counter} - {(frame_counter/args.fps):.2f}', (52,52), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
	cv2.putText(combo, f'{frame_counter} - {(frame_counter/args.fps):.2f}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
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


