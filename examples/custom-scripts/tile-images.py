#!/usr/bin/env python

import sys
import cv2
import math
import argparse
import datetime
import numpy as np
from pathlib import Path

from argument_parser import define_boolean_argument, var2opt

def tile(img, tile_size, offset, prefix):
	for i in range(int(math.ceil(img.shape[0]/(offset[1] * 1.0)))):
		for j in range(int(math.ceil(img.shape[1]/(offset[0] * 1.0)))):
			cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img.shape[1])]
			# Debugging the tiles
			cv2.imwrite(prefix + '-tiled-img-' + str(i) + "_" + str(j) + ".jpg", cropped_img)

parser = argparse.ArgumentParser()
parser.add_argument('--tile_size',	nargs='?', help='specify tile size in the format <w,h> (e.g. -tile_size 960,540)', required=True)
parser.add_argument('--offset',		nargs='?', help='specify offset in the format    <x,y> (e.g. -offset 960,540)', default='==tile_size')
parser.add_argument('--dir',		nargs='?', help="directory to search")
parser.add_argument('--img',		nargs='?', help="process a single image")
args = parser.parse_args()

tile_size = tuple(int(x) for x in str(args.tile_size).split(','))
offset    = tile_size if args.offset == '==tile_size' else tuple(int(x) for x in str(args.offset).split(','))

if args.dir is None and args.img is None:
	print('Please specify an image or a directory to scan for images. Exiting...')
	sys.exit(1)

if args.dir is not None:
	if not Path(args.dir).exists():
		print(f'Directory {args.dir} does not exist. Exiting...')
		sys.exit(1)

	print(f'{args.dir = }')
	search_path = Path(args.dir)

	for fn in search_path.glob('*.jpg'):
		img_fn = str(Path(fn).stem)
		print(f'Tiling img {img_fn} with size: {img.shape} - tile size: {tile_size} - offset: {offset}')
		img    = cv2.imread(str(fn))
		tile(img, tile_size, offset, img_fn)


if args.img is not None:
	img       = cv2.imread(str(args.img))
	print(f'Img size: {img.shape} - tile size: {tile_size} - offset: {offset}')
	tile(img, tile_size, offset, str(Path(args.img).stem))


#img = cv2.imread('/mnt/porcodiodo/datasets/ericsson-360/output/output-ericsson-camera-360-4k-frames-10fps/ericsson-camera-360-4k-frames-10fps-00818.png')


