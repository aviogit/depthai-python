#!/usr/bin/env python3

import depthai as dai
import argparse

# Note: OV9282 is capable of higher FPS, but those configs are not added yet:
# 1280x720 / 130fps
# 1280x800 / 120fps
#  640x400 / 210fps  
options = {
    1: (1280, 720, 60, dai.MonoCameraProperties.SensorResolution.THE_720_P),
    2: (1280, 800, 60, dai.MonoCameraProperties.SensorResolution.THE_800_P),
    3: (640, 400, 120, dai.MonoCameraProperties.SensorResolution.THE_400_P),
}
print("Options:")
for opt in options: print(opt, ":", options[opt])

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--opt', type=int, default=1, 
                    help="Select an option from above. Default: %(default)s")
parser.add_argument('-n', '--no-preview', default=False, action="store_true",
                     help="Disable preview, record to file only")
parser.add_argument('-r', '--rotate', default=False, action="store_true",
                     help="Rotate image 180 degrees")
parser.add_argument('-m', '--mirror', default=False, action="store_true",
                     help="Mirror image horizontally")
parser.add_argument('-f', '--flip', default=False, action="store_true",
                     help="Flip image vertically")
args = parser.parse_args()

option = args.opt
show_preview = not args.no_preview  # May increase system load when enabled

orientation = dai.CameraImageOrientation.NORMAL
if args.rotate: orientation = dai.CameraImageOrientation.ROTATE_180_DEG
if args.mirror: orientation = dai.CameraImageOrientation.HORIZONTAL_MIRROR
if args.flip:   orientation = dai.CameraImageOrientation.VERTICAL_FLIP

res_w, res_h, fps, res_desc = options.get(option)
print("Config:", res_w, 'x', res_h, '@', fps)

# Start defining a pipeline
pipeline = dai.Pipeline()

cam_list = ['left', 'right']
camera = {}
video_enc = {}
video_out = {}
preview_out = {}
for i in cam_list:
    camera[i] = pipeline.createMonoCamera()
    camera[i].setResolution(res_desc)
    camera[i].setFps(fps)
    camera[i].setImageOrientation(orientation)

    video_enc[i] = pipeline.createVideoEncoder()
    video_enc[i].setDefaultProfilePreset(res_w, res_h, fps,
                        dai.VideoEncoderProperties.Profile.H264_MAIN)
    camera[i].out.link(video_enc[i].input)

    video_out[i] = pipeline.createXLinkOut()
    video_out[i].setStreamName('video_' + i)
    video_enc[i].bitstream.link(video_out[i].input)

    if show_preview:
        preview_out[i] = pipeline.createXLinkOut()
        preview_out[i].setStreamName('preview_' + i)
        camera[i].out.link(preview_out[i].input)
camera['left'].setBoardSocket(dai.CameraBoardSocket.LEFT)
camera['right'].setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Pipeline defined, now the device is assigned and pipeline is started
dev = dai.Device(pipeline)
dev.startPipeline()

# Output queues will be used to get the encoded data from the outputs defined above
queue_video = {}
queue_preview = {}
file_name = {}
for i in cam_list:
    queue_video[i] = dev.getOutputQueue(video_out[i].getStreamName(),
                                        maxSize=30, blocking=True)
    file_name[i] = 'mono_' + i + '.h264'

    if show_preview:
        queue_preview[i] = dev.getOutputQueue(preview_out[i].getStreamName(),
                                              maxSize=3, blocking=False)
        import cv2

file = {}
with open(file_name['left'], 'wb') as file['left'], open(file_name['right'], 'wb') as file['right']:
    print("Press Ctrl+C to stop recording...")
    while True:
        try:
            for i in cam_list:
                # Empty each queue
                while queue_video[i].has():
                    queue_video[i].get().getData().tofile(file[i])
            if show_preview:
                for i in cam_list:
                    pkt = None
                    while queue_preview[i].has():
                        pkt = queue_preview[i].get()
                    if pkt is not None:
                        shape = (pkt.getHeight(), pkt.getWidth())
                        frame = pkt.getData().reshape(shape)
                        cv2.imshow(i, frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            break

print('\nSaved to disk:')
for f in file_name:
    print(file_name[f])
