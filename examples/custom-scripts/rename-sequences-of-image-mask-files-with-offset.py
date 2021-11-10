#!/usr/bin/env python3

import os
import glob
import argparse

from argument_parser import define_boolean_argument, var2opt

parser = argparse.ArgumentParser()
parser.add_argument('--offset', nargs='?', type=int, default=0, help="offset to be added/subtracted from the current filename")
define_boolean_argument(parser, *var2opt('dry_run'), 'just show the modifications, don\'t actually rename any file', True)
args = parser.parse_args()

# 044997.jpg

extension='.jpg'


flist = glob.glob('0*'+extension)
for i in flist:
	oldfname = i.replace(extension, '')		# 044997
	oldfbody = int(oldfname)			# 44997
	newfbody = oldfbody + args.offset		# 44990	(e.g. -7 offset)
	fprefix  = oldfname.replace(str(oldfbody), '')	# 0
	newfname = fprefix + str(newfbody) + extension	# 044990.jpg
	print(f'Renaming: {i} -> {newfname}')
	if not args.dry_run:
		os.rename(i, newfname)
