#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np

import subprocess as sp
import shlex

from pathlib import Path

from argument_parser import define_boolean_argument, var2opt

from utils import wlsFilter

debug = False

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', nargs='?', help="color-/left-/right-<prefix>.h265 video files will be used")
parser.add_argument('--start-frame', default=0, type=int, help='start frame for start replaying the video triplet')
parser.add_argument('--format', default='h265', help="encoding output file format")
define_boolean_argument(parser, *var2opt('disparity'), 'capture disparity instead of left/right streams', False)
define_boolean_argument(parser, *var2opt('wls_disparity'), 'capture wls disparity instead of left/right or normal disparity streams', True)
define_boolean_argument(parser, *var2opt('show_wls_preview'), 'show host-side WLS filtering made with OpenCV', False)
define_boolean_argument(parser, *var2opt('preview'), 'show images during disparity conversion', False)
args = parser.parse_args()

if args.prefix is None:
	print(f'Please specify a valid prefix for video files. Exiting...')
	sys.exit(0)


baseline = 75 #mm
disp_levels = 96
fov = 71.86

wlsFilter = wlsFilter(args, _lambda=8000, _sigma=1.5, baseline=baseline, fov=fov, disp_levels=disp_levels)

depth_cap  = cv2.VideoCapture(f'depth-{args.prefix}.h265')
depth_len  = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
rright_cap = cv2.VideoCapture(f'rectright-{args.prefix}.h265')
rright_len = int(rright_cap.get(cv2.CAP_PROP_FRAME_COUNT))

process = sp.Popen(shlex.split(f'ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 depth-{args.prefix}.h265'), stdout=sp.PIPE)
# Wait for sub-process to finish
process.wait()
line = process.stdout.readline()
ffprobe_nframes = int(line)

print(f'{rright_len = } - {depth_len = } - {ffprobe_nframes = }')

#small_size = (1280, 720)
small_size = (640, 400)

pause = False

if '265' in args.format:
	vcodec = 'libx265'
elif '264' in args.format: 
	vcodec = 'libx264'
elif 'mjpg' in args.format.lower() or 'mjpeg' in args.format.lower(): 
	vcodec = 'mjpeg'

wls_outfn = f'wls-{args.prefix}-{vcodec}.mp4'
depth_fps, depth_width, depth_height =  30, 1280, 720
#wls_cap = cv2.VideoWriter(wls_outfn, cv2.VideoWriter.fourcc('M','J','P','G'), depth_fps, (depth_width, depth_height))
#wls_cap = cv2.VideoWriter(wls_outfn, cv2.VideoWriter.fourcc('A','V','C','1'), depth_fps, (depth_width, depth_height))
#wls_cap = cv2.VideoWriter(wls_outfn, cv2.VideoWriter_fourcc(*'avc1'), depth_fps, (depth_width, depth_height))
#wls_cap = cv2.VideoWriter(wls_outfn, cv2.VideoWriter_fourcc(*'h264'), depth_fps, (depth_width, depth_height))
#wls_cap = cv2.VideoWriter(wls_outfn, 0x21, depth_fps, (depth_width, depth_height))
#wls_cap = cv2.VideoWriter(wls_outfn, cv2.VideoWriter_fourcc(*'mp4v'), depth_fps, (depth_width, depth_height))	# <--- this works (.mp4 container)
#wls_cap = cv2.VideoWriter(wls_outfn, cv2.VideoWriter_fourcc(*'h265'), depth_fps, (depth_width, depth_height))

'''
if not wls_cap.isOpened():
	print(f'Output cv2.VideoWriter is not open, probably for encoder errors. Exiting...')
	sys.exit(0)
'''
# Open ffmpeg application as sub-process
# FFmpeg input PIPE: RAW images in BGR color format
# FFmpeg output MP4 file encoded with HEVC codec.
# Arguments list:
# -y                   Overwrite output file without asking
# -s {width}x{height}  Input resolution width x height (1344x756)
# -pixel_format bgr24  Input frame color format is BGR with 8 bits per color component
# -f rawvideo          Input format: raw video
# -r {fps}             Frame rate: fps (25fps)
# -i pipe:             ffmpeg input is a PIPE
# -vcodec libx265      Video codec: H.265 (HEVC)
# -pix_fmt yuv420p     Output video color space YUV420 (saving space compared to YUV444)
# -crf 24              Constant quality encoding (lower value for higher quality and larger output file).
# {output_filename}    Output file name: output_filename (output.mp4)
process = sp.Popen(shlex.split(f'ffmpeg -y -s {depth_width}x{depth_height} -pixel_format bgr24 -f rawvideo -r {depth_fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -crf 24 {wls_outfn}'), stdin=sp.PIPE)


counter = 0
perc = prev_perc = 0

while depth_cap.isOpened() and rright_cap.isOpened():

	if args.preview:
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

	if debug:
		print(f'{dframe.shape} - {rrframe.shape} - {small_size}')

	disp_gray	= cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
	rr_img		= cv2.flip(rrframe, flipCode=1)
	wls_data_item   = counter, disp_gray, rr_img
	counter, filtered_disp, colored_disp = wlsFilter.apply_wls_filter(wls_data_item)
	colored_disp    = cv2.flip(colored_disp, 1)	# flip colored_disp horizontally because rectified_right is flipped
	combo		= np.concatenate((rr_img, colored_disp), axis=0)

	counter += 1

	#wls_cap.write(colored_disp)
	process.stdin.write(colored_disp.tobytes())

	prev_perc = perc
	perc = 10000.0 * counter / ffprobe_nframes
	if perc != prev_perc:
		print(f'Progress: {int(perc)/100.0}% ', end='')
	if debug:
		print(f'{counter = } - {perc = } - {prev_perc}')

	if args.preview:
		cv2.imshow('frame', combo)

depth_cap.release()
rright_cap.release()

cv2.destroyAllWindows()

# Close and flush stdin
process.stdin.close()
# Wait for sub-process to finish
process.wait()
# Terminate the sub-process
process.terminate()

sys.exit(0)


