#!/usr/bin/env python3

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

from utils import dequeue, dequeued_frames_dict, datetime_from_string, create_encoder, wlsFilter, apply_wls_filter

args = argument_parser()

# Run with:
# ./rgb-depth-encoding.py --output-dir /tmp --confidence 200 --no-extended-disparity --depth-resolution 720p --wls-filter

# Close to 30 FPS, no depth preview, no RGB, but it should be possible to do WLS filtering offline with just h264 depth and h264 rectified right
# ./rgb-depth-encoding.py --output-dir /tmp --confidence 200 --no-extended-disparity --depth-resolution 720p --rectified-right --no-write-preview --no-rgb


start_time		= datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
color_outfn		= f'{args.output_dir}/color-{start_time}.h264'
wls_outfn		= f'{args.output_dir}/wls-{start_time}.avi'
rr_outfn		= f'{args.output_dir}/rectright-{start_time}.h264'
if args.disparity:
	depth_outfn	= f'{args.output_dir}/depth-{start_time}.h264'
else:
	left_outfn	= f'{args.output_dir}/left-{start_time}.h264'
	right_outfn	= f'{args.output_dir}/right-{start_time}.h264'


# video = cv2.VideoWriter('appsrc ! queue ! videoconvert ! video/x-raw ! omxh264enc ! video/x-h264 ! h264parse ! rtph264pay ! udpsink host=192.168.0.2 port=5000 sync=false',0,25.0,(640,480))


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

'''
cam_rgb.initialControl.setManualFocus(130)

# This may be redundant when setManualFocus is used
cam_rgb.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF) 
'''


# Define a source - two mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dprofile)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dprofile)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)


# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(args.confidence)

#median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 # For depth filtering
#depth.setMedianFilter(median)

#depth.setExtendedDisparity(args.extended_disparity)
#depth.setOutputRectified(True)		# The rectified streams are horizontally mirrored by default
#depth.setOutputDepth(False)
depth.setRectifyEdgeFillColor(0)	# Black, to better see the cutout from rectification (black stripe on the edges)
depth.setLeftRightCheck(False)

# Normal disparity values range from 0..95, will be used for normalization
max_disparity = 95

if args.extended_disparity:
	max_disparity *= 2 # Double the range
depth.setExtendedDisparity(args.extended_disparity)

if args.subpixel_disparity:
	max_disparity *= 32 # 5 fractional bits, x32
depth.setSubpixel(args.subpixel_disparity)

# When we get disparity to the host, we will multiply all values with the multiplier
# for better visualization
multiplier = 255 / max_disparity



if args.wls_filter or args.rectified_right:
	xoutRectifiedRight = pipeline.createXLinkOut()
	xoutRectifiedRight.setStreamName("rectifiedRight")
	depth.rectifiedRight.link(xoutRectifiedRight.input)
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



wlsFilter = wlsFilter(args, _lambda=8000, _sigma=1.5)

baseline = 75 #mm
disp_levels = 96
fov = 71.86


'''
class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print str(e)
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        return ret, bgr
'''

args_websocket = False
if args_websocket:
	#host_port = ('zapp-brannigan.ge.imati.cnr.it', 58889)
	host_port = ('localhost', 58889)
	clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	clientsocket.connect(host_port )


args_gstreamer = False
if args_gstreamer:
	fourcc = cv2.VideoWriter_fourcc(*'H264')
	#gstout = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency noise-reduction=10000 bitrate=2048 speed-preset=superfast ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=58889', fourcc, 30, (1280, 720), True)
	# ouput GStreamer pipeline
	#gstout = cv2.VideoWriter('appsrc ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=58889', fourcc, 30, (1280, 720), True)
	gstout = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency noise-reduction=10000 bitrate=2048 speed-preset=superfast ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=58889', fourcc, 30, (1280, 720), True)
	if not gstout.isOpened():
		print('VideoWriter not opened')
		sys.exit(0)



def compute_fps(curr_time, last_time, dequeued_frames_dict):
	if curr_time != last_time:
		#print(f'{curr_time = }')
		last_time = curr_time
		curr_time = datetime.now()
		run_time = curr_time - datetime_from_string(start_capture_time)
		display_str = ''
		for stream, frames in dequeued_frames_dict.items():
			if run_time.total_seconds() == 0:
				break
			microseconds = run_time.seconds * 1000000 + run_time.microseconds
			fps = frames*1000000/microseconds
			if display_str != '':
				display_str += ' - '
			display_str += stream + ' ' + str(frames) + ' ' + f'{fps:.2f}'
		print(display_str)
	return last_time




if (args.show_preview or args.write_preview) and args.disparity:
	# Create output
	#xout_rgb = pipeline.createXLinkOut()
	xout_dep = pipeline.createXLinkOut()
	#xout_rgb.setStreamName("rgb")
	xout_dep.setStreamName("disparity")
	
	if args.debug_pipeline_types:
		print(f'{type(cam_rgb) = } - {cam_rgb = }')
		print(f'{type(cam_rgb.video) = } - {cam_rgb.video = }')
		print(f'{type(cam_rgb.preview) = } - {cam_rgb.preview = }')
		print(f'{type(depth) = } - {depth = }')
		print(f'{type(depth.disparity) = } - {depth.disparity = }')
	
	#cam_rgb.preview.link(xout_rgb.input)
	depth.disparity.link(xout_dep.input)
	
	#cam_rgb.video.link(xout_rgb.input)
	#depth.video.link(xout_dep.input)




if args.rgb:
	videorgbEncoder,   videorgbOut  = create_encoder(pipeline, cam_rgb.video,        color_resolution, 'h264_rgb')
if args.disparity:
	videodispEncoder,  videodispOut	= create_encoder(pipeline, depth.disparity,      depth_resolution, 'h264_depth')
else:
	videoleftEncoder,  videoleftOut	= create_encoder(pipeline, depth.syncedLeft,     depth_resolution, 'h264_left')
	videorightEncoder, videorightOut= create_encoder(pipeline, depth.syncedRight,    depth_resolution, 'h264_right')
if args.wls_filter or args.rectified_right:
	videorrEncoder,    videorrOut   = create_encoder(pipeline, depth.rectifiedRight, depth_resolution, 'h264_rr')




if args.show_preview:
	cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('disparity', int(color_width/2), int(color_height/2))

	cv2.namedWindow('disparity th',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('disparity th', int(color_width/2), int(color_height/2))

	cv2.namedWindow('rgb',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('rgb', int(color_width/2), int(color_height/2))

	cv2.namedWindow('combo',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('combo', int(color_width/2), int(color_height/2))

# Pipeline defined, now the device is connected to
with dai.Device(pipeline, usb2Mode=args.force_usb2) as device:
	# Start pipeline
	device.startPipeline()

	if (args.show_preview or args.write_preview) and args.disparity:
		# Output queue will be used to get the rgb frames from the output defined above
		#q_rgb  = device.getOutputQueue(name="rgb",		maxSize=4,	blocking=False)
		q_dep  = device.getOutputQueue(name="disparity",	maxSize=4,	blocking=False)

	# Output queue will be used to get the encoded data from the output defined above
	if args.rgb:
		q_264c = device.getOutputQueue(name="h264_rgb",		maxSize=30,	blocking=False)
	if args.disparity:
		q_264d = device.getOutputQueue(name="h264_depth",	maxSize=30,	blocking=False)
	else:
		q_264l = device.getOutputQueue(name="h264_left",	maxSize=30,	blocking=False)
		q_264r = device.getOutputQueue(name="h264_right",	maxSize=30,	blocking=False)

	if args.wls_filter or args.rectified_right:
		q_rright = device.getOutputQueue(name="rectifiedRight",	maxSize=4,	blocking=False)
		q_264rr  = device.getOutputQueue(name="h264_rr",	maxSize=30,	blocking=False)


	cmap_counter = 0

	# The .h264 file is a raw stream file (not playable yet)
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

	print("Press Ctrl+C to stop encoding...")
	try:
		start_capture_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		last_time = start_time
		while True:
			if (args.show_preview or args.write_preview) and args.disparity:
				#in_rgb   = dequeue(q_rgb, 'rgb-preview'  , args, 1, debug=False)
				in_depth  = dequeue(q_dep, 'depth-preview', args, 2, debug=False)
			if args.rgb:
				in_h264c = dequeue(q_264c, 'rgb-h264'     , args, 3, debug=False)
			if args.disparity:
				in_h264d  = dequeue(q_264d, 'depth-h264'  , args, 4, debug=False)
			else:
				in_h264l  = dequeue(q_264l, 'left-h264'   , args, 5, debug=False)
				in_h264r  = dequeue(q_264r, 'right-h264'  , args, 6, debug=False)
			if args.wls_filter or args.rectified_right:
				in_h264rr = dequeue(q_264rr, 'rright-h264', args, 7, debug=False)
			if args.debug_pipeline_steps:
				print('8. all queues done')

			if args.rgb:
				in_h264c.getData().tofile(videorgbFile)		# appends the packet data to the opened file
			if args.disparity:
				in_h264d.getData().tofile(videodepthFile)	# appends the packet data to the opened file
			else:
				in_h264l.getData().tofile(videoleftFile)	# appends the packet data to the opened file
				in_h264r.getData().tofile(videorightFile)	# appends the packet data to the opened file
			if args.wls_filter or args.rectified_right:
				in_h264rr.getData().tofile(videorrFile)		# appends the packet data to the opened file
				'''
				rr_nv12 = in_h264rr.getData()
				rr_nv12.tofile(videorrFile)			# appends the packet data to the opened file
				print(type(rr_nv12), len(rr_nv12))
				'''
				if args_websocket:
					data   = pickle.dumps(rr_nv12)
					header = struct.pack("I", len(data))
					print(f'{header = }')
					clientsocket.sendall(header+data)	# https://stackoverflow.com/a/30988516/1396334
					#clientsocket.sendall(data)	# https://stackoverflow.com/a/30988516/1396334
				if args_gstreamer:
					gstout.write(rr_nv12)
				'''
				tmpfn = 'rright-h264-' + str(dequeued_frames_dict['rright-h264']) + '.raw'
				with open(tmpfn,'wb') as tmpfile:
					print(tmpfn)
					tmpfile.write(rr_nv12)
				'''

			curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
			last_time = compute_fps(curr_time, last_time, dequeued_frames_dict)
			'''
				print(f'{start_time = } - {start_capture_time = } - {curr_time = } - {run_time = }')
				print('Frames statistics:')
				for stream, frames in dequeued_frames_dict.items():
					fps = frames/run_time.total_seconds()
					print(f'{stream = } - {frames = } - {fps = :.2f}')
			'''
			'''
				frame = in_depth.getFrame()
				#print(f'{frame.shape = }')
				frame = (frame*multiplier).astype(np.uint8)
				frame = cv2.medianBlur(frame, 7)
				frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
				frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
				#cv2.imshow("disparity", frame)
				#if cv2.waitKey(1) == ord('q'):
				#	break
				cv2.imwrite('/tmp/depth.png', frame) 
			'''

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

			if args.wls_filter:
				in_rright = q_rright.get()
				rr_img    = in_rright.getFrame()
				rr_img    = cv2.flip(rr_img, flipCode=1)
				disp_img  = in_depth.getFrame()
				filtered_disp, colored_disp = apply_wls_filter(wlsFilter, disp_img, rr_img, baseline, fov, disp_levels, args)
				if args.write_wls_preview:
					wls_cap.write(colored_disp)
				if args.show_wls_preview:
					if cv2.waitKey(1) == ord('q'):
						break

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
	print("To view the encoded data, convert the stream file (.h264) into a video file (.mp4) using a command below:")
	print("ffmpeg -framerate 30 -i video.h264 -c copy video.mp4")

