
#!/usr/bin/env python3
import cv2
import depthai as dai
import threading, queue
import random
import time

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(1280, 720)
camRgb.setInterleaved(True)
camRgb.setColorOrder(dai.ColorCamera.Properties.ColorOrder.BGR)
camRgb.setFps(60)

# Create output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

running = True
guiQueue = queue.Queue()
def threadFunc(dev):

    x = random.randint(0,10)
    print('Waiting for: ', x, ' seconds.')
    y = time.sleep(x)

    print(f"{dev.getMxId()} {dev.state}")
    found, device_info = dai.Device.getDeviceByMxId(dev.getMxId())

    # Connect to the device
    with dai.Device(pipeline, device_info) as device:
        device.startPipeline()

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while running:
            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
            frm = inRgb.getCvFrame()
            cv2.putText(frm, str(inRgb.getTimestamp().total_seconds()), (15,15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
            guiQueue.put((dev.getMxId(), frm))
            #print(dev.getMxId(), '- Sent frame to GUI thread to be displayed.')

threads = []
for device in dai.Device.getAllAvailableDevices():
    thread = threading.Thread(target=threadFunc, args=[device])
    thread.start()
    threads.append(thread)

# GUI thread
while True:
    #print('Waiting for gui event...')
    ev = guiQueue.get()
    cv2.imshow(ev[0], ev[1])
    if cv2.waitKey(1) == ord('q'):
        break


# Join all threads
running = False
for t in threads:
    t.join()