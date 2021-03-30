#!/usr/bin/env python3

import cv2
import sys
import time
import depthai as dai
import numpy as np
from pathlib import Path


def apply_colormap(frame, cmap=0):
	if cmap == 0 or cmap > 21:
		return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
	if cmap == 1:
		return cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
	if cmap == 2:
		return cv2.applyColorMap(frame, cv2.COLORMAP_AUTUMN)
	if cmap == 3:
		return cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)
	if cmap == 4:
		return cv2.applyColorMap(frame, cv2.COLORMAP_RAINBOW)
	if cmap == 5:
		return cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
	if cmap == 6:
		return cv2.applyColorMap(frame, cv2.COLORMAP_SUMMER)
	if cmap == 7:
		return cv2.applyColorMap(frame, cv2.COLORMAP_SPRING)
	if cmap == 8:
		return cv2.applyColorMap(frame, cv2.COLORMAP_COOL)
	if cmap == 9:
		return cv2.applyColorMap(frame, cv2.COLORMAP_HSV)
	if cmap == 10:
		return cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
	if cmap == 11:
		return cv2.applyColorMap(frame, cv2.COLORMAP_PINK)
	if cmap == 12:
		return cv2.applyColorMap(frame, cv2.COLORMAP_PARULA)
	if cmap == 13:
		return cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)
	if cmap == 14:
		return cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
	if cmap == 15:
		return cv2.applyColorMap(frame, cv2.COLORMAP_PLASMA)
	if cmap == 16:
		return cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
	if cmap == 17:
		return cv2.applyColorMap(frame, cv2.COLORMAP_CIVIDIS)
	if cmap == 18:
		return cv2.applyColorMap(frame, cv2.COLORMAP_TWILIGHT)
	if cmap == 19:
		return cv2.applyColorMap(frame, cv2.COLORMAP_TWILIGHT_SHIFTED)
	if cmap == 20:
		return cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)
	if cmap == 21:
		return cv2.applyColorMap(frame, cv2.COLORMAP_DEEPGREEN)



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


cw, ch, cfps, cprofile = color_resolutions['1080p']
dw, dh, dfps, dprofile = depth_resolutions[ '400p']

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(cw, ch)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(cprofile)
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



# Create output
xout_rgb = pipeline.createXLinkOut()
xout_dep = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
xout_dep.setStreamName("disparity")
cam_rgb.preview.link(xout_rgb.input)
depth.disparity.link(xout_dep.input)

cam_rgb.video.link(xout_rgb.input)
#depth.video.link(xout_dep.input)




# Create an encoder, consuming the frames and encoding them using H.265 encoding
videoEncoder = pipeline.createVideoEncoder()
videoEncoder.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
#videoEncoder.setDefaultProfilePreset(3840, 2160, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(videoEncoder.input)

# Create output
videoOut = pipeline.createXLinkOut()
videoOut.setStreamName('h265')
videoEncoder.bitstream.link(videoOut.input)






# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
	# Start pipeline
	device.startPipeline()

	# Output queue will be used to get the rgb frames from the output defined above
	q_rgb = device.getOutputQueue(name="rgb",       maxSize=4,	blocking=False)
	q_dep = device.getOutputQueue(name="disparity", maxSize=4,	blocking=False)
	# Output queue will be used to get the encoded data from the output defined above
	q_265 = device.getOutputQueue(name="h265",	maxSize=30,	blocking=True)

	cmap_counter = 0

	# The .h265 file is a raw stream file (not playable yet)
	with open('color.h265','wb') as videoFile:
		print("Press Ctrl+C to stop encoding...")
		try:
			while True:
				in_rgb   = q_rgb.get()	# blocking call, will wait until a new data has arrived
				in_depth = q_dep.get()	# blocking call, will wait until a new data has arrived
				in_h265  = q_265.get()	# blocking call, will wait until a new data has arrived
				in_h265.getData().tofile(videoFile)	# appends the packet data to the opened file
		
				# data is originally represented as a flat 1D array, it needs to be converted into HxW form
				frame = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
				frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
				frame = np.ascontiguousarray(frame)
				# frame is transformed, the color map will be applied to highlight the depth info
				frame = apply_colormap(frame, cmap=13)
				# frame is ready to be shown
				cv2.imshow("disparity", frame)
		
				# Retrieve 'bgr' (opencv format) frame
				cv2.imshow("bgr", in_rgb.getCvFrame())
		
				if cv2.waitKey(1) == ord('q'):
					break

				cmap_counter += 1

		except KeyboardInterrupt:
			# Keyboard interrupt (Ctrl + C) detected
			pass

	print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
	print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")


sys.exit(0)

















'''
# Create encoder to produce JPEG images
video_enc = pipeline.createVideoEncoder()
video_enc.setDefaultProfilePreset(cam_rgb.getVideoSize(), cam_rgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
cam_rgb.video.link(video_enc.input)

# Create JPEG output
xout_jpeg = pipeline.createXLinkOut()
xout_jpeg.setStreamName("jpeg")
video_enc.bitstream.link(xout_jpeg.input)
'''


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the rgb frames from the output defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
    q_jpeg = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

    # Make sure the destination path is present before starting to store the examples
    Path('06_data').mkdir(parents=True, exist_ok=True)

    while True:
        in_rgb = q_rgb.tryGet()  # non-blocking call, will return a new data that has arrived or None otherwise

        if in_rgb is not None:
            # data is originally represented as a flat 1D array, it needs to be converted into HxW form
            shape = (in_rgb.getHeight() * 3 // 2, in_rgb.getWidth())
            frame_rgb = cv2.cvtColor(in_rgb.getData().reshape(shape), cv2.COLOR_YUV2BGR_NV12)
            # frame is transformed and ready to be shown
            cv2.imshow("rgb", frame_rgb)

        for enc_frame in q_jpeg.tryGetAll():
            with open(f"06_data/{int(time.time() * 10000)}.jpeg", "wb") as f:
                f.write(bytearray(enc_frame.getData()))

        if cv2.waitKey(1) == ord('q'):
            break




















