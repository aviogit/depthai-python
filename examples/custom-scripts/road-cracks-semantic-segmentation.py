#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
from datetime import datetime, timedelta

'''
Blob taken from the great PINTO zoo

git clone git@github.com:PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/026_mobile-deeplabv3-plus/01_float32/
./download.sh
source /opt/intel/openvino/bin/setupvars.sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py   --input_model deeplab_v3_plus_mnv2_decoder_256.pb   --model_name deeplab_v3_plus_mnv2_decoder_256   --input_shape [1,256,256,3]   --data_type FP16   --output_dir openvino/256x256/FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -ip U8 -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6 -m openvino/256x256/FP16/deeplab_v3_plus_mnv2_decoder_256.xml -o deeplabv3p_person_6_shaves.blob
'''

parser = argparse.ArgumentParser()
parser.add_argument("-shape", "--nn_shape", help="select NN model shape", default=256, type=int)
#parser.add_argument("-nn", "--nn_path", help="select model path for inference", default='models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob', type=str)
#parser.add_argument("-nn", "--nn_path", help="select model path for inference", default='models/pothole-segmentron-deeplabv3+-resnet50-no-data-aug-img_size-360-640-2b-2021-08-06_04.49.43-WD-0.0001-BS-8-LR-1e-07-1e-06-epoch-5-dice_multi-0.8379.blob', type=str)
parser.add_argument("-nn", "--nn_path", help="select model path for inference", default='models/pothole-segmentron-deeplabv3+-mobilenet_v2-basic-data-aug-img_size-360-640-2b-2021-08-06_13.07.29-WD-0.0001-BS-8-LR-1e-07-1e-06-epoch-1-dice_multi-0.8192.blob', type=str)
args = parser.parse_args()

#nn_shape = args.nn_shape
if '256' in args.nn_path:
	nn_img_size = (256,256)
else:
	nn_img_size = (640,360)
nn_path = args.nn_path
#TARGET_SHAPE = (400,400)

def dilate(src, dilatation_size, dilation_shape=cv2.MORPH_ELLIPSE):
	element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
	dilatation_dst = cv2.dilate(src, element)
	return dilatation_dst

def decode_deeplabv3p(output_tensor, new_shape, debug=False):
	class_colors = [[0,0,0], [0,0,255], [255,0,0], [0,255,0], [255,255,0], [0,255,255]]
	class_colors = np.asarray(class_colors, dtype=np.uint8)

	result_mask_ir = np.squeeze(np.argmax(output_tensor, axis=1)).astype(np.uint8)
	if debug:
		print(f'{result_mask_ir.shape} - {result_mask_ir = }')

	output_colors = np.take(class_colors, result_mask_ir, axis=0)

	output_colors = dilate(output_colors, 7)

	if debug:
		print(f'{output_colors.shape} - {output_colors = }')
		print(f'{np.nonzero(output_colors) = }')

	return output_colors

def get_multiplier(output_tensor, new_shape):
	class_binary = [[0], [1]]
	class_binary = np.asarray(class_binary, dtype=np.uint8)
	output = output_tensor.reshape(new_shape)
	output_colors = np.take(class_binary, output, axis=0)
	return output_colors

def show_deeplabv3p(output_colors, frame):
	return cv2.addWeighted(frame,1, output_colors,0.5,0)

def dispay_colored_depth(frame, name):
	frame_colored = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
	frame_colored = cv2.equalizeHist(frame_colored)
	frame_colored = cv2.applyColorMap(frame_colored, cv2.COLORMAP_HOT)
	cv2.imshow(name, frame_colored)
	return frame_colored

class FPSHandler:
	def __init__(self, cap=None):
		self.timestamp = time.time()
		self.start = time.time()
		self.frame_cnt = 0
	def next_iter(self):
		self.timestamp = time.time()
		self.frame_cnt += 1
	def fps(self):
		return self.frame_cnt / (self.timestamp - self.start)

class HostSync:
	def __init__(self):
		self.arrays = {}
	def add_msg(self, name, msg):
		if not name in self.arrays:
			self.arrays[name] = []
		self.arrays[name].append(msg)
	def get_msgs(self, timestamp):
		ret = {}
		for name, arr in self.arrays.items():
			for i, msg in enumerate(arr):
				time_diff = abs(msg.getTimestamp() - timestamp)
				# 20ms since we add rgb/depth frames at 30FPS => 33ms. If
				# time difference is below 20ms, it's considered as synced
				if time_diff < timedelta(milliseconds=20):
					ret[name] = msg
					self.arrays[name] = arr[i:]
					break
		return ret


def crop_to_square(frame):
	height = frame.shape[0]
	width  = frame.shape[1]
	delta = int((width-height) / 2)
	# print(height, width, delta)
	return frame[0:height, delta:width-delta]

# Start defining a pipeline
pipeline = dai.Pipeline()

if '2021.2' in args.nn_path:
	pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)
else:
	pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_3)

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# Color cam: 1920x1080
# Mono cam: 640x400
cam.setIspScale(2,3) # To match 400P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.initialControl.setManualFocus(130)


# For deeplabv3
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#cam.setPreviewSize(int(nn_shape/2), int(nn_shape/2))
cam.setPreviewSize(nn_img_size)
cam.setInterleaved(False)

# NN output linked to XLinkOut
isp_xout = pipeline.createXLinkOut()
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)


# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
cam.preview.link(detection_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

xout_passthrough = pipeline.createXLinkOut()
xout_passthrough.setStreamName("pass")
# Only send metadata, we are only interested in timestamp, so we can sync
# depth frames with NN output
# xout_passthrough.setMetadataOnly(True)
detection_nn.passthrough.link(xout_passthrough.input)

debug = False

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
	# Output queues will be used to get the outputs from the device
	q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
	#q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
	q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
	q_pass = device.getOutputQueue(name="pass", maxSize=4, blocking=False)

	fps = FPSHandler()
	sync = HostSync()
	depth_frame = None

	while True:
		'''
		in_depth = q_depth.tryGet()
		if in_depth is not None:
			sync.add_msg("depth", in_depth)
		'''

		in_nn = q_nn.tryGet()
		if in_nn is not None:
			fps.next_iter()
			# Get NN passthrough frame
			frame = q_pass.get().getCvFrame()

			if debug:
				print(f'{in_nn = }')
				all_layer_names = in_nn.getAllLayerNames()
				print(f'{all_layer_names = }')
				all_layers = in_nn.getAllLayers()
				print(f'{all_layers = }')
				print(f'{all_layers[0].dataType = }')
				print(f'{all_layers[0].dims = }')
				print(f'{all_layers[0].name = }')
				print(f'{all_layers[0].numDimensions = }')
				print(f'{all_layers[0].offset = }')
				print(f'{all_layers[0].order = }')
				print(f'{all_layers[0].strides = }')

			# get layer1 data
			if 'deeplab_v3_plus_mvn2_decoder_256' in args.nn_path:
				layer1 = in_nn.getFirstLayerInt32()
				new_shape = nn_img_size
			else:
				layer1 = in_nn.getFirstLayerFp16()
				#new_shape = all_layers[0].dims
				new_shape = (1, 3, 360, 640)
			lenlayer1 = len(layer1)

			if debug:
				print(f'{lenlayer1 = }')

			if lenlayer1 == 0:
				continue
			# reshape to numpy array
			lay1 = np.asarray(layer1, dtype=np.int32).reshape(new_shape)
			output_colors = decode_deeplabv3p(lay1, new_shape)

			# To match depth frames
			output_colors = cv2.resize(output_colors, nn_img_size)

			frame = show_deeplabv3p(output_colors, frame)
			cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
			cv2.imshow("weighted", frame)

		if cv2.waitKey(1) == ord('q'):
			break
