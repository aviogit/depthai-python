import cv2
import math
import numpy as np
import depthai as dai
from datetime import datetime


def datetime_from_string(str_time):
	return datetime.strptime(str_time, '%Y-%m-%d-%H-%M-%S')

dequeued_frames_dict = dict()
def dequeue(queue, name, args, dbg_step=0, debug=False):
	if name in dequeued_frames_dict:
		dequeued_frames_dict[name] += 1
	else:
		dequeued_frames_dict[name] = 1
	if debug or args.debug_pipeline_steps:
		curr_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')
		print(f'{dbg_step}. {curr_time}')
	pkt = queue.get()	# blocking call, will wait until a new data has arrived
	return pkt


class wlsFilter:
	wlsStream = "wlsFilter"

	'''
	def on_trackbar_change_lambda(self, value):
		self._lambda = value * 100
	def on_trackbar_change_sigma(self, value):
		self._sigma = value / float(10)
	'''

	def __init__(self, args, _lambda, _sigma, baseline, fov, disp_levels):
		self._lambda     = _lambda
		self._sigma      = _sigma
		self.baseline    = baseline
		self.fov         = fov
		self.disp_levels = disp_levels
		self.args        = args

		self.wlsFilter   = cv2.ximgproc.createDisparityWLSFilterGeneric(False)

		self.debug       = False

		if args.show_wls_preview:
			cv2.namedWindow(self.wlsStream)
		'''
		self.lambdaTrackbar = trackbar('Lambda', self.wlsStream, 0, 255, 80, self.on_trackbar_change_lambda)
		self.sigmaTrackbar  = trackbar('Sigma',  self.wlsStream, 0, 100, 15, self.on_trackbar_change_sigma)
		'''

	def filter(self, disparity, right, depthScaleFactor):
		# https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
		self.wlsFilter.setLambda(self._lambda)
		# https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
		self.wlsFilter.setSigmaColor(self._sigma)
		filteredDisp = self.wlsFilter.filter(disparity, right)

		# Compute depth from disparity (32 levels)
		with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
			# raw depth values
			depthFrame = (depthScaleFactor / filteredDisp).astype(np.uint16)

		return filteredDisp, depthFrame
	def apply_wls_filter(self, wls_data_item):
		if self.debug:
			print(type(wls_data_item))
		counter, disp_img, rr_img = wls_data_item
		print(f'Running apply_wls_filter() thread for item no.: {counter} and imgs: {disp_img.shape} - {rr_img.shape}')
		focal = disp_img.shape[1] / (2. * math.tan(math.radians(self.fov / 2)))
		depth_scale_factor = self.baseline * focal
	
		filtered_disp, depth_frame = self.filter(disp_img, rr_img, depth_scale_factor)
	
		if self.args.show_wls_preview:
			cv2.imshow("wls raw depth", depth_frame)
	
		filtered_disp = (filtered_disp * (255/(self.disp_levels-1))).astype(np.uint8)
		if self.args.show_wls_preview:
			cv2.imshow(wlsFilter.wlsStream, filtered_disp)
	
		colored_disp = cv2.applyColorMap(filtered_disp, cv2.COLORMAP_HOT)
		if self.args.show_wls_preview:
			cv2.imshow("wls colored disp", colored_disp)
	
		return counter, filtered_disp, colored_disp

def create_encoder(pipeline, source, profile_tuple, stream_name):
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

'''
def apply_wls_filter(wlsFilter, disp_img, r_img, baseline, fov, disp_levels, args):
	focal = disp_img.shape[1] / (2. * math.tan(math.radians(fov / 2)))
	depth_scale_factor = baseline * focal

	filtered_disp, depth_frame = wlsFilter.filter(disp_img, r_img, depth_scale_factor)

	if args.show_wls_preview:
		cv2.imshow("wls raw depth", depth_frame)

	filtered_disp = (filtered_disp * (255/(disp_levels-1))).astype(np.uint8)
	if args.show_wls_preview:
		cv2.imshow(wlsFilter.wlsStream, filtered_disp)

	colored_disp = cv2.applyColorMap(filtered_disp, cv2.COLORMAP_HOT)
	if args.show_wls_preview:
		cv2.imshow("wls colored disp", colored_disp)

	return filtered_disp, colored_disp
'''
