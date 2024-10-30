import depthai as dai
from text_extraction.Reader import Reader
import cv2

# pipeline
pipeline = dai.Pipeline()

# camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(1920,1080)

# XLinkOut
xoutCam = pipeline.create(dai.node.XLinkOut)
xoutCam.setStreamName('video')

# Link camera and output
cam.video.link(xoutCam.input)

reader = Reader()

def main():
    # main
    with dai.Device(pipeline) as device:

        queue = device.getOutputQueue('video')
        queue.setBlocking(False)
        queue.setMaxSize(1)

        while True:
            if cv2.waitKey(1) == ord('q'):
                break
            frame = queue.get().getCvFrame() # type: ignore
            reader.read(frame)
            reader.draw_predictions(stop=False)
            
        

if __name__ == '__main__':
    main()

        