#!/usr/bin/env python3

import os
import cv2
import sys
import time
import depthai as dai
import numpy as np

import socket
import struct
import pickle

from pathlib import Path

from datetime import datetime

from argument_parser import argument_parser

from colormaps import apply_colormap

from utils import dequeue, dequeued_frames_dict, datetime_from_string, create_encoder, wlsFilter, compute_fps

import multiprocessing

import colored					# https://pypi.org/project/colored/
rst = colored.attr('reset')

from circular_buffer import CircularBuffer
ringbuffer = CircularBuffer(10)

'''
def col(a, c):
	basic_colors = {
				'r': 'red'
				'g': 'green'
				'b': 'blue'
				'y': 'yellow'
			}
	if c in basic_colors:
		return colored(a, basic_colors[c])
	else
		return colored(a, c)
'''

def preview_thread_impl(args, preview_counter, depth_frame, disp_img, rr_img, colored_disp):
	'''
	# data is originally represented as a flat 1D array, it needs to be converted into HxW form
	depth_h, depth_w = in_depth.getHeight(), in_depth.getWidth()
	if args.debug_img_sizes:
		print(f'{depth_h = } - {depth_w = }')
	depth_frame = in_depth.getData().reshape((depth_h, depth_w)).astype(np.uint8)
	'''
	if args.preview_downscale_factor != 1:
		depth_frame = cv2.resize(depth_frame, dsize=(depth_w//args.preview_downscale_factor, depth_h//args.preview_downscale_factor), interpolation=cv2.INTER_CUBIC)
	if args.debug_img_sizes:
		print(f'{depth_frame.shape = } - {len(depth_frame) = } - {type(depth_frame) = } - {depth_frame.size = }')
	depth_frame_orig = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)

	if args.show_colored_disp:
		depth_frame = np.ascontiguousarray(depth_frame_orig)
		# depth_frame is transformed, the color map will be applied to highlight the depth info
		depth_frame = apply_colormap(depth_frame, cmap=13)
		# depth_frame is ready to be shown
		cv2.imshow("colored disparity", depth_frame)
		
	# Retrieve 'bgr' (opencv format) frame
	if args.show_rgb:
		rgb_frame = in_rgb.getCvFrame()
		if args.preview_downscale_factor != 1:
			rgb_frame = cv2.resize(rgb_frame, dsize=(color_width//args.preview_downscale_factor, color_height//args.preview_downscale_factor), interpolation=cv2.INTER_CUBIC)
		if args.debug_img_sizes:
			print(f'{rgb_frame.shape = } - {len(rgb_frame) = } - {type(rgb_frame) = } - {rgb_frame.size = }')
		cv2.imshow("rgb", rgb_frame)

	#img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if args.show_th_disp:
		depth_frame_th = cv2.adaptiveThreshold(depth_frame_orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		cv2.imshow("disparity th", depth_frame_th)
		depth_frame_th_color = cv2.cvtColor(depth_frame_th, cv2.COLOR_GRAY2BGR)

	if False:
		rgb_frame_resized = cv2.resize(rgb_frame, dsize=(depth_w, depth_h), interpolation=cv2.INTER_CUBIC)
		combo = (rgb_frame_resized + depth_frame_th_color) / 2
		cv2.imshow("combo", combo)

	if args.show_gray_disp and disp_img is not None:
		cv2.imshow("grayscale disparity",	disp_img)
	if args.show_rr_img and rr_img is not None:
		cv2.imshow("rr_img",			rr_img)
	if args.show_wls_preview:
		cv2.imshow("WLS colored disp",		colored_disp)

	if cv2.waitKey(1) == ord('q'):			# this is the culprit! https://answers.opencv.org/question/52774/waitkey1-timing-issues-causing-frame-rate-slow-down-fix/
		return
args = argument_parser()

# Run with:
# ./rgb-depth-encoding.py --confidence 200 --no-extended-disparity --depth-resolution 720p --wls-filter
# ./rgb-depth-encoding-v2.py --confidence 200 --no-extended-disparity --depth-resolution 720p --rectified-left --rectified-right --no-write-preview --no-rgb --wls-filter

# ./rgb-depth-encoding.py --confidence 200 --no-extended-disparity --depth-resolution 720p --rectified-right --no-write-preview --no-rgb

# v2 - 60 FPS with rectified-right + rectified-left + disparity
# ./rgb-depth-encoding-v2.py --confidence 200 --no-extended-disparity --depth-resolution 720p --rectified-left --rectified-right --no-write-preview --no-rgb

# v2 - 46.5 FPS with rectified-right + disparity + RGB 1080p
# ./rgb-depth-encoding-v2.py --confidence 200 --no-extended-disparity --depth-resolution 720p --rectified-right --no-write-preview



start_time		= datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
color_outfn		= f'{args.output_dir}/color-{start_time}.h265'
wls_outfn		= f'{args.output_dir}/wls-{start_time}.avi'
rr_outfn		= f'{args.output_dir}/rectright-{start_time}.h265'
rl_outfn		= f'{args.output_dir}/rectleft-{start_time}.h265'
if args.disparity:
	depth_outfn	= f'{args.output_dir}/depth-{start_time}.h265'
else:
	left_outfn	= f'{args.output_dir}/left-{start_time}.h265'
	right_outfn	= f'{args.output_dir}/right-{start_time}.h265'


# video = cv2.VideoWriter('appsrc ! queue ! videoconvert ! video/x-raw ! omxh265enc ! video/x-h265 ! h265parse ! rtph265pay ! udpsink host=192.168.0.2 port=5000 sync=false',0,25.0,(640,480))


# Start defining a pipeline
pipeline = dai.Pipeline()


color_resolutions = {
		'1080p': (1920,	1080, 60, dai.ColorCameraProperties.SensorResolution.THE_1080_P),
		'4K'   : (3840,	2160, 60, dai.ColorCameraProperties.SensorResolution.THE_4_K),
}
depth_resolutions = {
		'720p': (1280,	720, 60,  dai.MonoCameraProperties.SensorResolution.THE_720_P),
		'800p': (1280,	800, 60,  dai.MonoCameraProperties.SensorResolution.THE_800_P),
		'400p': (640,	400, 120, dai.MonoCameraProperties.SensorResolution.THE_400_P),
}

color_resolution = color_resolutions[args.color_resolution]
depth_resolution = depth_resolutions[args.depth_resolution]

color_width, color_height, color_fps, color_profile	= color_resolution
depth_width, depth_height, depth_fps, dprofile		= depth_resolution

if args.rgb:
	# Define a source - color camera
	cam_rgb = pipeline.createColorCamera()
	cam_rgb.setPreviewSize(color_width, color_height)
	cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
	cam_rgb.setResolution(color_profile)
	cam_rgb.setInterleaved(False)
	cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
	cam_rgb.setFps(color_fps)



print(f'\nSaving capture files to: {colored.fg("red") + args.output_dir + rst}\n')

'''
cam_rgb.initialControl.setManualFocus(130)

# This may be redundant when setManualFocus is used
cam_rgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF) 
'''


# Define a source - two mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dprofile)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(depth_fps)

right = pipeline.createMonoCamera()
right.setResolution(dprofile)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(depth_fps)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(args.confidence)

#depth.setOutputRectified(True)		# The rectified streams are horizontally mirrored by default
depth.setRectifyEdgeFillColor(0)	# Black, to better see the cutout from rectification (black stripe on the edges)
depth.setLeftRightCheck(False)

if args.wls_filter: #or args.rectified_right or args.rectified_left:
	xoutRectifiedRight = pipeline.createXLinkOut()
	xoutRectifiedRight.setStreamName("rectifiedRight")
	depth.rectifiedRight.link(xoutRectifiedRight.input)
	xoutRectifiedLeft = pipeline.createXLinkOut()
	xoutRectifiedLeft.setStreamName("rectifiedLeft")
	depth.rectifiedLeft.link(xoutRectifiedLeft.input)
	if args.write_wls_preview:
		wls_cap = cv2.VideoWriter(wls_outfn, cv2.VideoWriter.fourcc('M','J','P','G'), depth_fps, (depth_width, depth_height))
		#cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))


'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.
Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''
'''
# Better handling for occlusions:
depth.setLeftRightCheck(False)
# Closer-in minimum depth, disparity range is doubled:
depth.setExtendedDisparity(False)
# Better accuracy for longer distance, fractional disparity 32-levels:
depth.setSubpixel(False)
'''

left.out.link(depth.left)
right.out.link(depth.right)


baseline = 75 #mm
disp_levels = 96
fov = 71.86

# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
wls_data    = []	# filtered_disp, colored_disp = pool.map(apply_wls_filter, (disp_imgs, rr_imgs))
#wls_results = []
wls_counter = 0

def wls_worker(queue_in, queue_out, wlsFilter):
	print(f'wls_worker() thread {os.getpid()} starting...')
	while True:
		if args.debug_wls_threading:
			print(f'Thread {os.getpid()} dequeuing (wls_queue_in.size: {wls_queue_in.qsize()})...')
		item = queue_in.get(True)
		if args.debug_wls_threading:
			print(f'Thread {os.getpid()} got item {type(item)}...')
		wls_counter, disp_img, rr_img = item
		if args.debug_wls_threading:
			print(f'Thread {os.getpid()} got frame no: {wls_counter} - {disp_img.shape} - {rr_img.shape}...')
		wls_data = (wls_counter, disp_img, rr_img)
		wls_counter_out, filtered_disp, colored_disp = wlsFilter.apply_wls_filter(wls_data)
		if args.debug_wls_threading:
			print(f'Thread {os.getpid()} completed frame no: {wls_counter} ({wls_counter_out}) - {disp_img.shape} - {rr_img.shape}...')
		if args.write_wls_preview:
			wls_cap.write(colored_disp)
		if args.debug_wls_threading:
			print(f'Thread {os.getpid()} enqueuing (wls_queue_out.size: {wls_queue_out.qsize()}) frame no: {wls_counter} ({wls_counter_out}) - {disp_img.shape} - {rr_img.shape} - {filtered_disp.shape} - {colored_disp.shape}...')
		wls_queue_out.put((wls_counter_out, filtered_disp, colored_disp))
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
preview_data    = []
preview_counter = 0

def preview_worker(queue_in):
	print(f'preview_worker() thread {os.getpid()} starting...')
	while True:
		if args.debug_preview_threading:
			print(f'Thread {os.getpid()} dequeuing (preview_queue_in.size: {preview_queue_in.qsize()})...')
		item = queue_in.get(True)
		if args.debug_preview_threading:
			print(f'Thread {os.getpid()} got item {type(item)}...')
		preview_counter, depth_frame, disp_img, rr_img, colored_disp = item
		if args.debug_preview_threading:
			print(f'Thread {os.getpid()} got frame no: {preview_counter} - {depth_frame.shape}...')

		preview_thread_impl(args, preview_counter, depth_frame, disp_img, rr_img, colored_disp)

		'''
		preview_data = (preview_counter, disp_img, rr_img)
		preview_counter_out, filtered_disp, colored_disp = previewFilter.apply_preview_filter(preview_data)
		if args.debug_preview_threading:
			print(f'Thread {os.getpid()} completed frame no: {preview_counter} ({preview_counter_out}) - {disp_img.shape} - {rr_img.shape}...')
		if args.write_preview_preview:
			preview_cap.write(colored_disp)
		if args.debug_preview_threading:
			print(f'Thread {os.getpid()} enqueuing (preview_queue_out.size: {preview_queue_out.qsize()}) frame no: {preview_counter} ({preview_counter_out}) - {disp_img.shape} - {rr_img.shape} - {filtered_disp.shape} - {colored_disp.shape}...')
		preview_queue_out.put((preview_counter_out, filtered_disp, colored_disp))
		'''

# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
no_of_wls_threads = 16
wls_filter			= None
wls_queue_in			= None
wls_queue_out			= None
wls_th_pool			= None
if args.wls_filter:
	wls_filter		= wlsFilter(args, _lambda=8000, _sigma=1.5, baseline=baseline, fov=fov, disp_levels=disp_levels)
	wls_queue_in		= multiprocessing.Queue()
	wls_queue_out		= multiprocessing.Queue()
	wls_th_pool		= multiprocessing.Pool(no_of_wls_threads, wls_worker, (wls_queue_in, wls_queue_out, wls_filter, ))
	#                                                     don't forget the comma here  ^
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
no_of_preview_threads = 1
preview_filter			= None
preview_queue_in		= None
preview_th_pool			= None
if args.show_preview:
	preview_queue_in	= multiprocessing.Queue()
	preview_th_pool		= multiprocessing.Pool(no_of_preview_threads, preview_worker, (preview_queue_in, ))
	#                                                     don't forget the comma here  ^
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------





if args.wls_filter or args.show_preview:
	# Create output
	xout_dep = pipeline.createXLinkOut()
	xout_dep.setStreamName("disparity")
	
	if args.debug_pipeline_types:
		print(f'{type(cam_rgb) = } - {cam_rgb = }')
		print(f'{type(cam_rgb.video) = } - {cam_rgb.video = }')
		print(f'{type(cam_rgb.preview) = } - {cam_rgb.preview = }')
		print(f'{type(depth) = } - {depth = }')
		print(f'{type(depth.disparity) = } - {depth.disparity = }')
	
	depth.disparity.link(xout_dep.input)




if args.rgb:
	stream_name = 'h265_rgb'
	print(f'Enabling stream: {colored.fg("red") + stream_name + rst}')
	videorgbEncoder,   videorgbOut  = create_encoder(pipeline, cam_rgb.video,        color_resolution, stream_name)
if args.disparity:
	stream_name = 'h265_depth'
	print(f'Enabling stream: {colored.fg("blue") + stream_name + rst}')
	videodispEncoder,  videodispOut	= create_encoder(pipeline, depth.disparity,      depth_resolution, stream_name)
else:
	stream_name = 'h265_left'
	print(f'Enabling stream: {colored.fg("green") + stream_name + rst}')
	videoleftEncoder,  videoleftOut	= create_encoder(pipeline, depth.syncedLeft,     depth_resolution, stream_name)
	stream_name = 'h265_right'
	print(f'Enabling stream: {colored.fg("yellow") + stream_name + rst}')
	videorightEncoder, videorightOut= create_encoder(pipeline, depth.syncedRight,    depth_resolution, stream_name)
#if args.wls_filter or args.rectified_left:
if args.rectified_left:
	stream_name = 'h265_rl'
	print(f'Enabling stream: {colored.fg("light_green") + stream_name + rst}')
	videorlEncoder,    videorlOut   = create_encoder(pipeline, depth.rectifiedLeft,  depth_resolution, stream_name)
if args.wls_filter or args.rectified_right:
	stream_name = 'h265_rr'
	print(f'Enabling stream: {colored.fg("light_yellow") + stream_name + rst}')
	videorrEncoder,    videorrOut   = create_encoder(pipeline, depth.rectifiedRight, depth_resolution, stream_name)




if args.show_preview:
	depth_size = (depth_width, depth_height)
	color_size = (color_width, color_height)
	if args.preview_downscale_factor != 1:
		color_size = (color_width//args.preview_downscale_factor, color_height//args.preview_downscale_factor)
		depth_size = (depth_width//args.preview_downscale_factor, depth_height//args.preview_downscale_factor)

	'''
	if args.show_colored_disp:
		cv2.namedWindow('colored disparity',	cv2.WINDOW_NORMAL)
		cv2.resizeWindow('colored disparity',	depth_size)

	if args.show_th_disp:
		cv2.namedWindow('disparity th',		cv2.WINDOW_NORMAL)
		cv2.resizeWindow('disparity th',	depth_size)
	if args.show_gray_disp:
		cv2.namedWindow('grayscale disparity',	cv2.WINDOW_NORMAL)
		cv2.resizeWindow('grayscale disparity',	depth_size)

	if args.show_rgb:
		cv2.namedWindow('rgb',			cv2.WINDOW_NORMAL)
		cv2.resizeWindow('rgb',			color_size)

	if False:
		cv2.namedWindow('combo',		cv2.WINDOW_NORMAL)
		cv2.resizeWindow('combo',		color_size)

	if args.show_wls_preview:
		cv2.namedWindow('WLS colored disp',	cv2.WINDOW_NORMAL)
		cv2.resizeWindow('WLS colored disp',	depth_size)
	'''

'''
def slowdown(x):
	y = 1
	for i in range(1, x+1):
		y *= i
	return y
'''

#from slowdown import slowdown

# Pipeline defined, now the device is connected to
with dai.Device(pipeline, usb2Mode=args.force_usb2) as device:
	# Start pipeline
	#device.startPipeline()

	if args.wls_filter or args.show_preview:
		# Output queue will be used to get the rgb frames from the output defined above
		q_dep  = device.getOutputQueue(name="disparity",	maxSize=30,	blocking=False)

	# Output queue will be used to get the encoded data from the output defined above
	if args.rgb:
		q_265c = device.getOutputQueue(name="h265_rgb",		maxSize=30,	blocking=False)
	if args.disparity:
		q_265d = device.getOutputQueue(name="h265_depth",	maxSize=30,	blocking=False)
	else:
		q_265l = device.getOutputQueue(name="h265_left",	maxSize=30,	blocking=False)
		q_265r = device.getOutputQueue(name="h265_right",	maxSize=30,	blocking=False)

	if args.wls_filter:
		q_rright = device.getOutputQueue(name="rectifiedRight",	maxSize=30,	blocking=False)
		q_rleft  = device.getOutputQueue(name="rectifiedLeft",	maxSize=30,	blocking=False)
	if args.rectified_right:
		q_265rr  = device.getOutputQueue(name="h265_rr",	maxSize=30,	blocking=False)
	if args.rectified_left:
		q_265rl  = device.getOutputQueue(name="h265_rl",	maxSize=30,	blocking=False)


	cmap_counter = 0

	# The .h265 file is a raw stream file (not playable yet)
	if args.rgb:
		videorgbFile    = open(color_outfn,'wb')
	if args.disparity:
		#videorgbFile	= open(color_outfn,'wb')
		videodepthFile	= open(depth_outfn,'wb')
	else:
		#videorgbFile	= open(color_outfn,'wb')
		videoleftFile	= open(left_outfn, 'wb')
		videorightFile	= open(right_outfn,'wb')
	if args.wls_filter or args.rectified_right:
		videorrFile	= open(rr_outfn,   'wb')
	if args.wls_filter or args.rectified_left:
		videorlFile	= open(rl_outfn,   'wb')

	print("Press Ctrl+C to stop encoding...")
	try:
		start_capture_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		last_time = start_time

		while True:
			if args.rgb:
				in_h265c  = dequeue(q_265c,  'rgb-h265'   , args.debug_pipeline_steps, 1, debug=False)
			if args.wls_filter or args.rectified_right:
				in_h265rr = dequeue(q_265rr, 'rright-h265', args.debug_pipeline_steps, 2, debug=False)
			#if args.wls_filter or args.rectified_left:
			if args.rectified_left:
				in_h265rl = dequeue(q_265rl, 'rleft-h265' , args.debug_pipeline_steps, 3, debug=False)
			if args.disparity:
				in_h265d  = dequeue(q_265d,  'depth-h265' , args.debug_pipeline_steps, 4, debug=False)
			else:
				in_h265l  = dequeue(q_265l,  'left-h265'  , args.debug_pipeline_steps, 5, debug=False)
				in_h265r  = dequeue(q_265r,  'right-h265' , args.debug_pipeline_steps, 6, debug=False)
			if args.wls_filter or args.show_preview:
				in_depth  = dequeue(q_dep,   'depth-preview', args.debug_pipeline_steps, 7, debug=False)
			if args.debug_pipeline_steps:
				print('8. all queues done')

			if args.rgb:
				in_h265c.getData().tofile(videorgbFile)		# appends the packet data to the opened file
			if args.disparity:
				in_h265d.getData().tofile(videodepthFile)	# appends the packet data to the opened file
			else:
				in_h265l.getData().tofile(videoleftFile)	# appends the packet data to the opened file
				in_h265r.getData().tofile(videorightFile)	# appends the packet data to the opened file
			if args.wls_filter or args.rectified_right:
				in_h265rr.getData().tofile(videorrFile)		# appends the packet data to the opened file
			#if args.wls_filter or args.rectified_left:
			if args.rectified_left:
				in_h265rl.getData().tofile(videorlFile)		# appends the packet data to the opened file

			curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
			last_time = compute_fps(curr_time, last_time, start_capture_time, dequeued_frames_dict)

			#slowdown(100000)

			'''
			if args.show_preview:
				# data is originally represented as a flat 1D array, it needs to be converted into HxW form
				depth_h, depth_w = in_depth.getHeight(), in_depth.getWidth()
				if args.debug_img_sizes:
					print(f'{depth_h = } - {depth_w = }')
				depth_frame = in_depth.getData().reshape((depth_h, depth_w)).astype(np.uint8)
				if args.preview_downscale_factor != 1:
					depth_frame = cv2.resize(depth_frame, dsize=(depth_w//args.preview_downscale_factor, depth_h//args.preview_downscale_factor), interpolation=cv2.INTER_CUBIC)
				if args.debug_img_sizes:
					print(f'{depth_frame.shape = } - {len(depth_frame) = } - {type(depth_frame) = } - {depth_frame.size = }')
				depth_frame_orig = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)

				if args.show_colored_disp:
					depth_frame = np.ascontiguousarray(depth_frame_orig)
					# depth_frame is transformed, the color map will be applied to highlight the depth info
					depth_frame = apply_colormap(depth_frame, cmap=13)
					# depth_frame is ready to be shown
					cv2.imshow("colored disparity", depth_frame)
		
				# Retrieve 'bgr' (opencv format) frame
				if args.show_rgb:
					rgb_frame = in_rgb.getCvFrame()
					if args.preview_downscale_factor != 1:
						rgb_frame = cv2.resize(rgb_frame, dsize=(color_width//args.preview_downscale_factor, color_height//args.preview_downscale_factor), interpolation=cv2.INTER_CUBIC)
					if args.debug_img_sizes:
						print(f'{rgb_frame.shape = } - {len(rgb_frame) = } - {type(rgb_frame) = } - {rgb_frame.size = }')
					cv2.imshow("rgb", rgb_frame)

				#img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				if args.show_th_disp:
					depth_frame_th = cv2.adaptiveThreshold(depth_frame_orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
					cv2.imshow("disparity th", depth_frame_th)
					depth_frame_th_color = cv2.cvtColor(depth_frame_th, cv2.COLOR_GRAY2BGR)

				if False:
					rgb_frame_resized = cv2.resize(rgb_frame, dsize=(depth_w, depth_h), interpolation=cv2.INTER_CUBIC)
					combo = (rgb_frame_resized + depth_frame_th_color) / 2
					cv2.imshow("combo", combo)

				if cv2.waitKey(1) == ord('q'):			# this is the culprit! https://answers.opencv.org/question/52774/waitkey1-timing-issues-causing-frame-rate-slow-down-fix/
					break
			'''

			disp_img = rr_img = colored_disp = None
			if args.wls_filter:
				in_rright = q_rright.get()
				rr_img    = in_rright.getFrame()
				#rr_img    = cv2.flip(rr_img, flipCode=1)
				disp_img  = in_depth.getFrame()
				if args.preview_downscale_factor:
					rr_img   = cv2.resize(rr_img,   dsize=(depth_width//args.preview_downscale_factor, depth_height//args.preview_downscale_factor), interpolation=cv2.INTER_CUBIC)
					disp_img = cv2.resize(disp_img, dsize=(depth_width//args.preview_downscale_factor, depth_height//args.preview_downscale_factor), interpolation=cv2.INTER_CUBIC)

				if args.wls_max_queue == 0 or wls_queue_in.qsize() < args.wls_max_queue: 
					wls_counter += 1
					if args.debug_wls_threading:
						print(f'Main thread enqueuing frame no: {wls_counter} because wls_queue_in.size: {wls_queue_in.qsize()}...')
					#flipHorizontal = cv2.flip(rr_img, 1)
					wls_queue_in.put((wls_counter, disp_img, rr_img))
					'''
					if args.show_gray_disp:
						cv2.imshow("grayscale disparity",     disp_img)
					if args.show_rr_img:
						cv2.imshow("rr_img_flipH", rr_img)
					'''
				if args.show_wls_preview:
					if args.wls_max_queue == 0 or wls_queue_in.qsize() < args.wls_max_queue: 
						if args.debug_wls_threading:
							print(f'Main thread dequeuing frame because wls_queue_out.size: {wls_queue_out.qsize()}...')
						item = wls_queue_out.get(True)
						if args.debug_wls_threading:
							print(f'Main thread got item {type(item)}...')
						wls_counter_out, filtered_disp, colored_disp = item
						if args.debug_wls_threading:
							print(f'Main thread got frame no: {wls_counter_out} - {filtered_disp.shape} - {colored_disp.shape}...')
						#cv2.imshow("WLS colored disp", colored_disp)

						'''
						if True or args.write_wls_preview:
							wls_cap.write(colored_disp)
						'''

			if args.show_preview:
				if args.preview_max_queue == 0 or preview_queue_in.qsize() < args.preview_max_queue: 
					if args.debug_preview_threading:
						print(f'Main thread enqueuing frame no: {preview_counter} because preview_queue_in.size: {preview_queue_in.qsize()}...')
					# data is originally represented as a flat 1D array, it needs to be converted into HxW form
					depth_h, depth_w = in_depth.getHeight(), in_depth.getWidth()
					if args.debug_img_sizes:
						print(f'{depth_h = } - {depth_w = }')
					depth_frame = in_depth.getData().reshape((depth_h, depth_w)).astype(np.uint8)
					preview_queue_in.put((preview_counter, depth_frame, disp_img, rr_img, colored_disp))
					preview_counter += 1


			cmap_counter += 1

	except KeyboardInterrupt:
		# Keyboard interrupt (Ctrl + C) detected
		if args.disparity:
			videodepthFile.close()
		else:
			videoleftFile.close()
			videorightFile.close()

	run_time = datetime_from_string(curr_time) - datetime_from_string(start_capture_time)
	print(f'{start_time = } - {start_capture_time = } - {curr_time = } - {run_time = }')
	print('Frames statistics:')
	for stream, frames in dequeued_frames_dict.items():
		fps = frames/run_time.total_seconds()
		print(f'{stream = } - {frames = } - {fps = :.2f}')
	print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
	print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")

