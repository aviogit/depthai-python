#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-class1', nargs='?', help='class 1: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-dir',   nargs='?', help="directory to search", default='.')
parser.add_argument('-dry-run', type=eval, choices=[True, False], default='True', help='don\'t write anything, just show old and new masks')
args = parser.parse_args()

if not Path(args.dir).exists():
	print(f'Directory {args.dir} does not exist. Exiting...')
	sys.exit(1)

print(f'{args.dir = }')

search_path = Path(args.dir)

colors1 = args.class1.split('-')
from1   = [int(i) for i in colors1[0].split(',')]
to1     = [int(i) for i in colors1[1].split(',')]

print(f'Converting class1 masks from: {from1} to: {to1}')

print(f'Dry run flag is: {args.dry_run}')

for fn in search_path.glob('*.png'):
	print(f'Reading image {str(fn)}...')
	img  = cv2.imread(str(fn))

	mask_color_lo = np.array(from1)
	mask_color_hi = np.array(from1)

	cv2.imshow('Current mask', cv2.resize(img, (640, 360)))
	cv2.moveWindow('Current mask', 200, 200)
	cv2.waitKey(1)

	mask = cv2.inRange(img, mask_color_lo, mask_color_hi)
	img[mask>0] = tuple(to1)

	cv2.imshow('New mask',  cv2.resize(img, (640, 360)))
	cv2.moveWindow('New mask', 840, 200)
	cv2.waitKey(1)

	if not args.dry_run:
		print(f'Writing image {str(fn)}...')
		cv2.imwrite(str(fn), img)
