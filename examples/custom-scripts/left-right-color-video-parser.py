#!/usr/bin/env python

import sys
import cv2
import argparse
import datetime
import numpy as np
from pathlib import Path

color_cap  = cv2.VideoCapture('color-2021-05-10-16-26-53.h265')
left_cap   = cv2.VideoCapture('left-2021-05-10-16-26-53.h265')
right_cap  = cv2.VideoCapture('right-2021-05-10-16-26-53.h265')

small_size = (1280, 720)

pause = False

while color_cap.isOpened() and left_cap.isOpened() and right_cap.isOpened():

	delay_ms = 1 #if args.continuous else 0
	if pause:
		delay_ms = 100
	key = cv2.waitKey(delay_ms)
	if key & 0xFF == ord('q'):
		break
	if key & 0xFF == ord('p'):
		pause = not pause
	if pause:
		continue

	cret, cframe = color_cap.read()
	lret, lframe = left_cap.read()
	rret, rframe = right_cap.read()

	#print(f'{cframe.shape} - {lframe.shape} - {rframe.shape}')

	cframe   = cv2.resize(cframe, small_size)
	#print(f'{small_size[1]/4} - {lframe.shape[1]}')
	cframe_s = cframe[int(small_size[0]/4):cframe.shape[1], :]
	lframe_s = lframe[int(small_size[0]/4):lframe.shape[1], :]
	rframe_s = rframe[int(small_size[0]/4):rframe.shape[1], :]

	#print(f'{cframe_s.shape} - {lframe_s.shape} - {rframe_s.shape}')

	combo = np.concatenate((lframe_s, cframe_s), axis=0)
	combo = np.concatenate((combo,    rframe_s), axis=0)

	cv2.imshow('frame', combo)

color_cap.release()
cv2.destroyAllWindows()

sys.exit(0)


