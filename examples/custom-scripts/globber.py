#!/usr/bin/env python3

import glob

def globber(args_prefix, args_rectright, mp4_suffix=''):
	depth_fn = glob.glob(f'depth-{args_prefix}*h265{mp4_suffix}')
	if args_rectright:
		main_fn = glob.glob(f'rectright-{args_prefix}*h265{mp4_suffix}')
		if len(main_fn) == 0:
			main_fn  = glob.glob(f'*{args_prefix}-rectright*h265{mp4_suffix}')
			depth_fn = glob.glob(f'*{args_prefix}-disp*h265{mp4_suffix}')
		if len(main_fn) == 0:
			main_fn  = glob.glob(f'*{args_prefix}-rright*h265{mp4_suffix}')
	else:
		main_fn = glob.glob(f'color-{args_prefix}*h265{mp4_suffix}')
		if len(main_fn) == 0:
			main_fn  = glob.glob(f'*{args_prefix}-color*h265{mp4_suffix}')
			depth_fn = glob.glob(f'*{args_prefix}-disp*h265{mp4_suffix}')
	return main_fn, depth_fn
