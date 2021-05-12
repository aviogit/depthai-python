#!/usr/bin/env python3

import cv2
import sys
import time
import depthai as dai
import numpy as np
from pathlib import Path

from datetime import datetime

from argument_parser import argument_parser

from colormaps import apply_colormap

args = argument_parser()

start_time	= datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

color_outfn	= f'{args.output_dir}/color-{start_time}.h265'
#depth_outfn	= f'{args.output_dir}/depth-{start_time}.h265'
left_outfn	= f'{args.output_dir}/left-{start_time}.h265'
right_outfn	= f'{args.output_dir}/right-{start_time}.h265'




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


color_width, color_height, color_fps, color_profile	= color_resolutions['1080p']
depth_width, depth_height, depth_fps, dprofile		= depth_resolutions['720p']

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(color_width, color_height)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(color_profile)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Define a source - two mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dprofile)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dprofile)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)















# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
depth.setMedianFilter(median)

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



if args.show_preview:
	# Create output
	xout_rgb = pipeline.createXLinkOut()
	xout_dep = pipeline.createXLinkOut()
	xout_rgb.setStreamName("rgb")
	xout_dep.setStreamName("disparity")
	
	if args.debug_pipeline_types:
		print(f'{type(cam_rgb) = } - {cam_rgb = }')
		print(f'{type(cam_rgb.video) = } - {cam_rgb.video = }')
		print(f'{type(cam_rgb.preview) = } - {cam_rgb.preview = }')
		print(f'{type(depth) = } - {depth = }')
		print(f'{type(depth.disparity) = } - {depth.disparity = }')
	
	cam_rgb.preview.link(xout_rgb.input)
	depth.disparity.link(xout_dep.input)
	
	cam_rgb.video.link(xout_rgb.input)
	#depth.video.link(xout_dep.input)


def create_encoder(source, profile_tuple, stream_name):
	# Create an encoder, consuming the frames and encoding them using H.265 encoding
	encoder = pipeline.createVideoEncoder()
	w, h, fps, resolution = profile_tuple
	codec = dai.VideoEncoderProperties.Profile.H265_MAIN if '265' in stream_name else dai.VideoEncoderProperties.Profile.H264_MAIN
	encoder.setDefaultProfilePreset(w, h, fps, codec)
	source.link(encoder.input)

	# Create output
	output = pipeline.createXLinkOut()
	output.setStreamName(stream_name)
	encoder.bitstream.link(output.input)

	return encoder, output


videorgbEncoder,   videorgbOut   = create_encoder(cam_rgb.video,     color_resolutions['1080p'], 'h265_rgb')
#videodispEncoder,  videodispOut  = create_encoder(depth.disparity,   depth_resolutions['720p'],  'h265_depth')
videoleftEncoder,  videoleftOut  = create_encoder(depth.syncedLeft,  depth_resolutions['720p'],  'h265_left')
videorightEncoder, videorightOut = create_encoder(depth.syncedRight, depth_resolutions['720p'],  'h265_right')

'''
# Create an encoder, consuming the frames and encoding them using H.265 encoding
videorgbEncoder = pipeline.createVideoEncoder()
videorgbEncoder.setDefaultProfilePreset(color_width, color_height, color_fps, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(videorgbEncoder.input)

# Create output
videorgbOut = pipeline.createXLinkOut()
videorgbOut.setStreamName('h265_rgb')
videorgbEncoder.bitstream.link(videorgbOut.input)


# Create an encoder, consuming the frames and encoding them using H.265 encoding
videodispEncoder = pipeline.createVideoEncoder()
videodispEncoder.setDefaultProfilePreset(depth_width, depth_height, depth_fps, dai.VideoEncoderProperties.Profile.H265_MAIN)
depth.disparity.link(videodispEncoder.input)

# Create output
videodispOut = pipeline.createXLinkOut()
videodispOut.setStreamName('h265_depth')
videodispEncoder.bitstream.link(videodispOut.input)
'''



if args.show_preview:
	cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('disparity', int(color_width/2), int(color_height/2))

	cv2.namedWindow('disparity th',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('disparity th', int(color_width/2), int(color_height/2))

	cv2.namedWindow('rgb',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('rgb', int(color_width/2), int(color_height/2))

	cv2.namedWindow('combo',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('combo', int(color_width/2), int(color_height/2))

last_time = start_time

# Pipeline defined, now the device is connected to
with dai.Device(pipeline, usb2Mode=args.force_usb2) as device:
	# Start pipeline
	device.startPipeline()

	if args.show_preview:
		# Output queue will be used to get the rgb frames from the output defined above
		q_rgb  = device.getOutputQueue(name="rgb",	maxSize=4,	blocking=False)
		q_dep  = device.getOutputQueue(name="disparity",maxSize=4,	blocking=False)
	# Output queue will be used to get the encoded data from the output defined above
	q_265c = device.getOutputQueue(name="h265_rgb",		maxSize=30,	blocking=False)
	q_265l = device.getOutputQueue(name="h265_left",	maxSize=30,	blocking=False)
	q_265r = device.getOutputQueue(name="h265_right",	maxSize=30,	blocking=False)
	q_265d = None
	#q_265d = device.getOutputQueue(name="h265_depth",	maxSize=30,	blocking=True)

	cmap_counter = 0

	# The .h265 file is a raw stream file (not playable yet)
	#with open(color_outfn,'wb') as videorgbFile, open(left_outfn,'wb') as videoleftFile, open(right_outfn,'wb') as videorightFile, open(depth_outfn,'wb') as videodepthFile:
	with open(color_outfn,'wb') as videorgbFile, open(left_outfn,'wb') as videoleftFile, open(right_outfn,'wb') as videorightFile:
		print("Press Ctrl+C to stop encoding...")
		try:
			while True:
				if args.show_preview:
					if args.debug_pipeline_steps:
						print('1.')
					in_rgb   = q_rgb.get()	# blocking call, will wait until a new data has arrived
					if args.debug_pipeline_steps:
						print('2.')
					in_depth = q_dep.get()	# blocking call, will wait until a new data has arrived
				if args.debug_pipeline_steps:
					print('3.')
				in_h265c = q_265c.get()		# blocking call, will wait until a new data has arrived
				if args.debug_pipeline_steps:
					print('4.')
				in_h265l = q_265l.get()		# blocking call, will wait until a new data has arrived
				if args.debug_pipeline_steps:
					print('5.')
				in_h265r = q_265r.get()		# blocking call, will wait until a new data has arrived
				if args.debug_pipeline_steps:
					print('6.')
				#in_h265d = q_265d.get()		# blocking call, will wait until a new data has arrived
				if args.debug_pipeline_steps:
					print('7.')

				'''
				if args.debug_img_sizes:
					print(f'{type(in_h265c)} - {len(in_h265c)}')
					print(f'{type(in_h265d)} - {len(in_h265d)}')
				'''

				in_h265c.getData().tofile(videorgbFile)		# appends the packet data to the opened file
				in_h265l.getData().tofile(videoleftFile)	# appends the packet data to the opened file
				in_h265r.getData().tofile(videorightFile)	# appends the packet data to the opened file
				#in_h265d.getData().tofile(videodepthFile)	# appends the packet data to the opened file
				in_h265d = None

				curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
				if curr_time != last_time:
					print(f'{curr_time = }')
					last_time = curr_time

				if args.show_preview:
					# data is originally represented as a flat 1D array, it needs to be converted into HxW form
					depth_h, depth_w = in_depth.getHeight(), in_depth.getWidth()
					if args.debug_img_sizes:
						print(f'{depth_h = } - {depth_w = }')
					depth_frame = in_depth.getData().reshape((depth_h, depth_w)).astype(np.uint8)
					if args.debug_img_sizes:
						print(f'{depth_frame.shape = } - {len(depth_frame) = } - {type(depth_frame) = } - {depth_frame.size = }')
					depth_frame_orig = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
					depth_frame = np.ascontiguousarray(depth_frame_orig)
					# depth_frame is transformed, the color map will be applied to highlight the depth info
					depth_frame = apply_colormap(depth_frame, cmap=13)
					# depth_frame is ready to be shown
					cv2.imshow("disparity", depth_frame)
			
					# Retrieve 'bgr' (opencv format) frame
					rgb_frame = in_rgb.getCvFrame()
					if args.debug_img_sizes:
						print(f'{rgb_frame.shape = } - {len(rgb_frame) = } - {type(rgb_frame) = } - {rgb_frame.size = }')
					cv2.imshow("rgb", rgb_frame)
	
					#img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					depth_frame_th = cv2.adaptiveThreshold(depth_frame_orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
					cv2.imshow("disparity th", depth_frame_th)
					depth_frame_th_color = cv2.cvtColor(depth_frame_th, cv2.COLOR_GRAY2BGR)
	
					rgb_frame_resized = cv2.resize(rgb_frame, dsize=(depth_w, depth_h), interpolation=cv2.INTER_CUBIC)
					combo = (rgb_frame_resized + depth_frame_th_color) / 2
					cv2.imshow("combo", combo)
			
					if cv2.waitKey(1) == ord('q'):
						break

				cmap_counter += 1

		except KeyboardInterrupt:
			# Keyboard interrupt (Ctrl + C) detected
			pass

	print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
	print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")

