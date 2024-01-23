#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np
from pathlib import Path

from classes.argument_parser import define_boolean_argument, var2opt

# E.g. conversion for surface-pattern-recognition dataset:
# /mnt/porcodiodo/cnr/depthai/depthai-python/examples/custom-scripts/segmentation-mask-change-color.py -dir `pwd` -class1 255,255,255-0,0,0 -class2 255,0,0-1,1,1 -class3 0,0,255-2,2,2 -class4 0,255,0-3,3,3 -class5 44,0,0-4,4,4 -class6 185,26,255-5,5,5 -class7 0,211,255-6,6,6 -class8 0,88,0-7,7,7 --no-dry-run --to-grayscale

def bitwise_or_but_img1_overwrites_img2(img1, img2, color):
	# Performs betwise_or only after having converted one of the
	# two masks to black & white colors (always RGB, no grayscale).
	# After the bitwise_or, the white portion of the new mask is
	# converted back to the original color to restore the original
	# class of the mask. In this way, the converted (e.g. red->white)
	# mask overwrites (because it's an OR operation among image bits)
	# the other mask and then it re-gains its original color.

	from_c = [int(i) for i in color.split(',')]
	mask_color_lo = np.array(from_c)
	mask_color_hi = np.array(from_c)

	mask = cv2.inRange(img1, mask_color_lo, mask_color_hi)
	if np.all((mask == 0)) and not np.all((img1 == 0)):
		print(f'Warning! Query mask (with your color: {color}) is all black! Check the provided color, maybe you inverted it with the other mask!')
	img1[mask>0] = tuple((255,255,255))

	img3 = cv2.bitwise_or(img1, img2)

	mask_color_lo = np.array([255,255,255])
	mask_color_hi = np.array([255,255,255])
	mask = cv2.inRange(img3, mask_color_lo, mask_color_hi)
	img3[mask>0] = tuple(from_c)

	return img3

parser = argparse.ArgumentParser()
#parser.add_argument('-class20', nargs='?', help='class 1: convert from <b1,g1,r1> to <b1a,g1a,r1a> (e.g. -class1 255,255,255-0,0,255)')
parser.add_argument('-dir1',			nargs='?', help="directory to search", default='.')
parser.add_argument('-dir2',			nargs='?', help="directory to search", default='.')
parser.add_argument('-dir-out',			nargs='?', help="output directory", default='/tmp')
parser.add_argument('--replace-from-suffix',	nargs='?', help="when looking for filenames, replace this fn1 suffix with -replace-to-suffix to produce fn2 (e.g. '.png' -> '.jpg')", default='')
parser.add_argument('--replace-to-suffix',	nargs='?', help="when looking for filenames, replace the fn1 suffix into -replace-from-suffix match to produce fn2 (e.g. '.png' -> '.jpg')", default='')
parser.add_argument('-op',			nargs='?', help="operation to perform (choose from bitwise_or, bitwise_and, bitwise_xor)", default='bitwise_or')
parser.add_argument('-color',			nargs='?', help='specify the original color for "bitwise-or-but-img*-overwrites-img*" in <b,g,r> format (e.g. red is -color 0,0,255)')
define_boolean_argument(parser, *var2opt('to_grayscale'), 'convert to grayscale after color conversion, before writing to file'	, False)
define_boolean_argument(parser, *var2opt('dry_run')	, 'don\'t write anything, just show old and new masks'			, True)
args = parser.parse_args()

if not Path(args.dir1).exists():
	print(f'Directory {args.dir1} does not exist. Exiting...')
	sys.exit(1)
if not Path(args.dir2).exists():
	print(f'Directory {args.dir2} does not exist. Exiting...')
	sys.exit(1)
if Path(args.dir_out).exists():
	print(f'Directory {args.dir_out} already exists. Exiting...')
	sys.exit(1)
else:
	Path(args.dir_out).mkdir(parents=True, exist_ok=True)



print(f'{args.dir1 = } & {args.dir2 = } -> {args.dir_out = }')

search_path = Path(args.dir1)

'''
from_to = []

for arg in vars(args):
	#print(arg, getattr(args, arg))
	val = getattr(args, arg)
	if 'class' in arg and val is not None:
		#print(val)
		colors = val.split('-')
		from_c = [int(i) for i in colors[0].split(',')]
		to_c   = [int(i) for i in colors[1].split(',')]
		from_to.append((from_c, to_c))

print(f'Converting class masks in this way: {from_to = }')
'''

print(f'Dry run flag is: {args.dry_run}')

for fn in search_path.glob('*.png'):
	print(f'Reading image {str(fn)}...')
	img1  = cv2.imread(str(fn))
	fn2 = str(fn).replace(f'{args.dir1}','')
	if args.replace_from_suffix != '' and args.replace_to_suffix != '':
		fn2 = str(fn2).replace(f'{args.replace_from_suffix}', f'{args.replace_to_suffix}')
	fn2 = Path(args.dir2) / fn2
	print(f'Reading image {str(fn2)}...')
	img2  = cv2.imread(str(fn2))

	cv2.imshow('Mask 1', cv2.resize(img1, (640, 360)))
	cv2.moveWindow('Mask 1',   0, 200)
	cv2.imshow('Mask 2', cv2.resize(img2, (640, 360)))
	cv2.moveWindow('Mask 2', 640, 200)
	cv2.waitKey(1)

	'''
	for from_c, to_c in from_to:
		#print(from_c)
		#print(to_c)
		mask_color_lo = np.array(from_c)
		mask_color_hi = np.array(from_c)

		mask = cv2.inRange(img, mask_color_lo, mask_color_hi)
		img[mask>0] = tuple(to_c)
	'''

	if args.op == 'bitwise_or':
		img3 = cv2.bitwise_or(img1, img2)
	elif args.op == 'bitwise_and':
		img3 = cv2.bitwise_and(img1, img2)
	elif args.op == 'bitwise_xor':
		img3 = cv2.bitwise_xor(img1, img2)
	elif args.op == 'bitwise-or-but-img1-overwrites-img2':
		img3 = bitwise_or_but_img1_overwrites_img2(img1, img2, args.color)
	elif args.op == 'bitwise-or-but-img2-overwrites-img1':
		img3 = bitwise_or_but_img1_overwrites_img2(img2, img1, args.color)
	else:
		print('Unknown operation, exiting...')
		sys.exit(0)

	cv2.imshow('New mask', cv2.resize(img3, (640, 360)))
	cv2.moveWindow('New mask', 1280, 200)
	cv2.waitKey(1)

	if not args.dry_run:
		if args.to_grayscale:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		out_fn = str(fn).replace(f'{args.dir1}', '')
		if args.replace_from_suffix != '' and args.replace_to_suffix != '':
			out_fn = str(out_fn).replace(f'{args.replace_from_suffix}', '.png')
		out_fn = Path(args.dir_out) / out_fn
		print(f'Writing {"grayscale " if args.to_grayscale else ""}image {str(out_fn)}...')
		cv2.imwrite(str(out_fn), img3)
