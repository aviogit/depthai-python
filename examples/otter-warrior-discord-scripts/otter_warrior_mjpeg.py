import contextlib
import depthai as dai
import csv
import datetime
from time import time, monotonic

# Define constants ============================================================================

# Paths
out_prefix = "capture_"
# Cameras
fps = 30
# Maximum recording time (10 minutes = 600 s)
t_record_max = 600.0

# OAK1 Camera ==================================================================================

pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setFps(fps)
camRgb.setInterleaved(False)

# Set resolution
rgb_w_px = 3840
rgb_h_px = 2160
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

# Create an encoder, consuming the frames and encoding them using H.265 encoding
videoEncoder = pipeline.createVideoEncoder()
videoEncoder.setDefaultProfilePreset(rgb_w_px, rgb_h_px, fps, dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.video.link(videoEncoder.input)

# Create output
videoOut = pipeline.createXLinkOut()
videoOut.setStreamName('h265')
videoEncoder.bitstream.link(videoOut.input)

# Create empty lists
qRgb_list = []
output_path_list = []
videoFile_list = []
csvFile_list = []
csvWriter_list = []
n_oak = 0

with contextlib.ExitStack() as stack:
    # look for connected Depthai devices
    for device_info in dai.Device.getAllAvailableDevices():
        device = stack.enter_context(dai.Device(pipeline, device_info))
        print("Connected to " + device_info.getMxId())
        # start pipeline
        device.startPipeline()
        # create output filename
        output_path_list.append(out_prefix+str(n_oak))
        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="h265", maxSize=fps, blocking=False)
        qRgb_list.append(qRgb)
        n_oak += 1

    # for each OAK found, create H265 and csv file
    for i in range(0,n_oak):
        videoFile_list.append(open(output_path_list[i] + '.h265','wb'))
        csvFile_list.append(open(output_path_list[i]+".csv", mode='a'))
        csvWriter_list.append(csv.writer(csvFile_list[i], delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
        csvWriter_list[i].writerow(["Frame number", "Sequence number", "Timestamp (monotonic)", "Timestamp (system time)"])

    t_start = time()
    t_record_curr = 0.0
    num_frames = [0 for i in range(0,n_oak)]

    print("Press Ctrl+C to stop recording ...")

    try:
        # end recording at maximum time, if not stopped before
        while t_record_curr<t_record_max:

            for i, qRgb in enumerate(qRgb_list):
                h265Packet = qRgb.tryGet()

                if h265Packet is not None:
                    # appends the packet data to the opened file
                    h265Packet.getData().tofile(output_path_list[i] + '_' + str(h265Packet.getSequenceNum()) + '.jpeg')
                    # retrieve timestamps and write them to csv file
                    ts_monotonic = h265Packet.getTimestamp()
                    ts_time = ts_monotonic / datetime.timedelta(seconds=1) - monotonic() + time()
                    ts_time = datetime.datetime.fromtimestamp(ts_time)
                    seq_num = h265Packet.getSequenceNum()
                    csvWriter_list[i].writerow([num_frames[i], seq_num, h265Packet.getTimestamp(), ts_time])
                    # increase frame count
                    num_frames[i] += 1

            t_record_curr = time() - t_start

    except KeyboardInterrupt:
        # Keyboard interrupt (Ctrl + C) detected
        # close all open files
        for i in range(0,n_oak):
            if videoFile_list[i]:
                videoFile_list[i].close()
            if csvWriter_list[i]:
                csvFile_list[i].close()
        pass
