#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from time import sleep
import datetime
import argparse
from pathlib import Path

#datasetDefault = str((Path(__file__).parent / Path('models/dataset')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', nargs='?', help="Path to recorded frames", default='/mnt/btrfs-data/')
parser.add_argument('-sc', '--save-combo',		dest="save_combo",		action='store_true', default=False,	help="save combo view (L+disparity)")
parser.add_argument('-mb', '--median-blur',		dest="median_blur",		action='store_true', default=True,	help="apply median blur filter")
parser.add_argument('-at', '--adaptive-threshold',	dest="adaptive_threshold",	action='store_true', default=True,	help="apply adaptive thresholding")
args = parser.parse_args()

if not Path(args.dataset).exists():
	print(f'Directory {args.dataset} does not exist. Exiting...')
	#raise FileNotFoundError(f'Dataset not found...')
	sys.exit(1)

print(f'{args.dataset = }')	# e.g. /media/biagio/btrfs-data/04-save-synced-frames-20210421-175210

dataset_path = Path(args.dataset)

subdirs = [x for x in dataset_path.iterdir() if x.is_dir()]

#print(subdirs)
print(f'Found {len(subdirs)} subdirectories')

small_size = (960, 540)

for d in subdirs:
	print(f'Processing directory: {d}')
	#color     = cv2.imread(str(d/'color.png'))
	left      = cv2.imread(str(d/'left.png'))
	disparity = cv2.imread(str(d/'disparity.png'))
	if args.median_blur:
		disparity = cv2.medianBlur(disparity, 7)
	if args.adaptive_threshold:
		disp_gray = cv2.cvtColor(disparity, cv2.COLOR_BGR2GRAY)
		disp_grth = cv2.adaptiveThreshold(disp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		disp_gray = cv2.cvtColor(disp_grth, cv2.COLOR_GRAY2BGR)

		disp_grth = cv2.medianBlur(disp_grth, 7)
		edge_gray = cv2.Canny(disp_grth, 100, 200)
		edges     = cv2.cvtColor(edge_gray, cv2.COLOR_GRAY2BGR)
		#cv2.morphologyEx()
		#cv2.dilate()

	#cv2.imshow('color', color)
	#cv2.imshow('disparity', disparity)
	#color_s = cv2.resize(color, small_size)
	left_s   = cv2.resize(left,      small_size)
	disp_s   = cv2.resize(disparity, small_size)
	disp_ths = cv2.resize(disp_gray, small_size)
	edges_s  = cv2.resize(edges,     small_size)
	combo1 = np.concatenate((left_s,   disp_s),  axis=1)
	combo2 = np.concatenate((disp_ths, edges_s), axis=1)
	combo  = np.concatenate((combo1, combo2),    axis=0)

	cv2.imshow('combo', combo)
	cv2.waitKey(100)
	if args.save_combo:
		outfn = dataset_path/str('combo-' + d.name + '.png')
		cv2.imwrite(str(outfn), combo)

sys.exit(0)
