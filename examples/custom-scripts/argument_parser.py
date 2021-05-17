import sys
import argparse
from distutils.util import strtobool

def define_boolean_argument(parser, var_name, cmd_name, dst_variable, help_str, default, debug = False):
	parser.add_argument('--'+cmd_name, dest=dst_variable, action='store_true', help=help_str)
	parser.add_argument('--no-'+cmd_name, dest=dst_variable, action='store_false')
	#parser.set_defaults(show_label=True)
	cmd_fstring = f'parser.set_defaults({var_name}={bool(strtobool(str(default)))})'
	if debug:
		print(cmd_fstring)
	exec(cmd_fstring)
	if debug:
		print(f'{var_name = } - {bool(strtobool(str(default))) = }')
		#exec("print(%s)" % var_name)

def var2opt(dst_variable):
	return dst_variable, dst_variable.replace('_', '-'), dst_variable

def argument_parser():

	parser = argparse.ArgumentParser(description='OAK-D video and depth h265 capture script')

	# ---------------------
	# -- CAPTURE OPTIONS --
	# ---------------------
	define_boolean_argument(parser, *var2opt('disparity'),			'capture disparity instead of left/right streams'	, True)
	#define_boolean_argument(parser, *var2opt('leftright'),			'capture left/right instead of disparity stream'	, False)
	parser.add_argument('--confidence',  type=int, default=250.0,	help="set the confidence treshold for disparity")

	# ------------------
	# -- OUTPUT FILES --
	# ------------------
	parser.add_argument('--output-dir', default='/mnt/btrfs-data',		help='captured videos output directory')

	# --------------
	# -- HARDWARE --
	# --------------
	define_boolean_argument(parser, *var2opt('force_usb2'), 'force the OAK-D camera in USB2 mode (useful in low bitrate/low power scenarios)', False)

	# ---------------
	# -- DEBUGGING --
	# ---------------
	define_boolean_argument(parser, *var2opt('debug_img_sizes'),		'add debugging information about captured image sizes'	, False)
	define_boolean_argument(parser, *var2opt('debug_pipeline_types'),	'add debugging information about captured image types'	, False)
	define_boolean_argument(parser, *var2opt('debug_pipeline_steps'),	'add debugging information about capturing steps'	, False)

	# ------------------
	# -- VIEW OPTIONS --
	# ------------------
	define_boolean_argument(parser, *var2opt('show_preview'),		'show OpenCV windows with the captured images'		, False)

	'''
	parser.set_defaults(show_fps=False)
	parser.set_defaults(show_frame_number=False)
	parser.set_defaults(debug_segments=False)
	parser.set_defaults(enable_ros=False)
	parser.set_defaults(batch_mode=False)
	parser.set_defaults(double_view=False)
	parser.set_defaults(demo_mode=False)
	parser.set_defaults(replay_mode=False)
	parser.set_defaults(do_inference=False)
	parser.set_defaults(do_inference_v2_seq=False)

	parser.set_defaults(show_only_scatter_points=False)
	parser.set_defaults(show_nodes=True)
	parser.set_defaults(show_axis=False)
	parser.set_defaults(show_color=True)
	parser.set_defaults(show_tips_traces=True)
	parser.set_defaults(show_edge_traces=False)
	'''

	args = parser.parse_args()
	print('')
	print('')
	print(f'Python   received this arguments: {sys.argv}')
	print('')
	print(f'Argparse received this arguments: {args}')

	return args
