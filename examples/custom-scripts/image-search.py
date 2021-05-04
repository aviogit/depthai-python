#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-query', nargs='?', help="Image to be searched")
parser.add_argument('-dir',   nargs='?', help="Directory to search", default='/tmp')
parser.add_argument('-mse-th',  type=float, default=50.0, help="MSE threshold below which it is possible to say that the difference of two images is only noise due to JPEG compression")
parser.add_argument('-psnr-th', type=float, default=30.0, help="PSNR threshold above which it is possible to say that the difference of two images is only noise due to JPEG compression")
args = parser.parse_args()

if not Path(args.dir).exists():
	print(f'Directory {args.dir} does not exist. Exiting...')
	sys.exit(1)

print(f'{args.dir = }')

search_path = Path(args.dir)

query = cv2.imread(args.query)

print(f'Using the following thresholds: {args.mse_th = } - {args.psnr_th = }')

cv2.imshow('Query', cv2.resize(query, (640, 360)))
cv2.waitKey(1000)

def condition(mse, psnr, mse_th, psnr_th):
	return mse < mse_th or psnr > psnr_th

for fn in search_path.glob('*.jpg'):
	img  = cv2.imread(str(fn))
	diff = query - img
	mse  = (np.square(query - img)).mean()
	cv2.imshow('Current Image', cv2.resize(img, (640, 360)))
	cv2.imshow('Current Diff',  cv2.resize(diff, (640, 360)))
	psnr = cv2.PSNR(query, img)
	if condition(mse, psnr, args.mse_th, args.psnr_th):
		print(80*'-')
		print(80*'-')
		print(80*'-')
	print(f'{fn = } - {mse = } - {psnr = }')
	if condition(mse, psnr, args.mse_th, args.psnr_th):
		print(80*'-')
		print(80*'-')
		print(80*'-')
		cv2.waitKey(3600000)
	cv2.waitKey(1)
