#!/usr/bin/env python

from exif import Image as ExifImage
from PIL import Image
from glob import glob
import sys
import os.path

import argparse

from argument_parser import define_boolean_argument, var2opt

parser = argparse.ArgumentParser()
parser.add_argument('--dir', nargs='?', help="working directory")
define_boolean_argument(parser, *var2opt('rotate_only_orientation_1_by_180'), 'don\'t ask me why, but the crack500 dataset seems to have wrong masks only for 1 (normal top-left) orientation', False)
args = parser.parse_args()

if args.dir is None:
        print(f'Please specify a valid working directory. Exiting...')
        sys.exit(0)

# https://www.impulseadventure.com/photo/exif-orientation.html
# https://jdhao.github.io/2019/07/31/image_rotation_exif_info/
transformation_funcs = {
    1: lambda img: img.rotate(180, resample=Image.BICUBIC, expand=True),		# used only if rotate-only-orientation-1-by-180 is True
    6: lambda img: img.rotate(-90, resample=Image.BICUBIC, expand=True),
    8: lambda img: img.rotate(90, resample=Image.BICUBIC, expand=True),
    3: lambda img: img.rotate(180, resample=Image.BICUBIC, expand=True),
    2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    5: lambda img: img.rotate(-90, resample=Image.BICUBIC, expand=True).transpose(Image.FLIP_LEFT_RIGHT),
    7: lambda img: img.rotate(90, resample=Image.BICUBIC, expand=True).transpose(Image.FLIP_LEFT_RIGHT),
    4: lambda img: img.rotate(180, resample=Image.BICUBIC, expand=True).transpose(Image.FLIP_LEFT_RIGHT),
}


for img_path in glob(os.path.join(args.dir, '*.jpg')):
	print(img_path)
	with open(img_path, 'rb') as image_file:
		my_image = ExifImage(image_file)
	try:
		orientation = my_image.orientation.value
	except:
		print('  N/A skipping')
		continue
	if orientation == 1:
		if not args.rotate_only_orientation_1_by_180:
			print('  1 skipping')
			continue
		else:
			print(' ', orientation, 'transforming (180° rotation)')
	else:
		if not args.rotate_only_orientation_1_by_180:
			print(' ', orientation, 'transforming')
		else:
			print(' ', orientation, ', but skipping because of --rotate-only-1 flag')
			continue
		
	my_image.orientation = 1
	with open(img_path, 'wb') as image_file:
		image_file.write(my_image.get_file())
		
	img = Image.open(img_path)
	img = transformation_funcs[orientation](img)
	img.save(img_path)
