import cv2
import numpy as np
import depthai as dai
import east
from utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog='Gerwazy')

    parser.add_argument('-p', '--preview', choices=[0, 1, 2], default=0, 
                        help=' - 0:\n\tno preview\n - 1:\n\tpreview window with image from camera and bounding boxes for texts\n - 2:\n\t1 option + additional window with regions with texts cut out and stacked on top of each other', type=int)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print additional info to the console')

    return parser.parse_args()


def main(args):
    pipeline = create_pipeline()
    with dai.Device(pipeline) as device:

        # Video queue
        q_vid = device.getOutputQueue("video", 4, blocking=False)
        # Detections queue
        # This should be set to block, but would get to some extreme queuing/latency!
        q_det = device.getOutputQueue("detections", 4, blocking=False)
        # Recognitions queue
        q_rec = device.getOutputQueue("recognitions", 4, blocking=True)
        q_manip_img = device.getInputQueue("manip_img")
        q_manip_cfg = device.getInputQueue("manip_cfg")
        q_manip_out = device.getOutputQueue("manip_out", 4, blocking=False)

        controlQueue = device.getInputQueue('control')

        frame = None
        cropped_stacked = None
        rotated_rectangles = []
        rec_pushed = 0
        rec_received = 0
        host_sync = HostSeqSync()

        characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
        codec = CTCCodec(characters)

        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        ctrl.setAutoFocusTrigger()
        controlQueue.send(ctrl)

        while True:
            vid_in = q_vid.tryGet()

            if vid_in is not None:
                host_sync.add_msg(vid_in)

            texts = []

            # Multiple recognition results may be available, read until queue is empty
            while True:
                in_rec = q_rec.tryGet()

                if in_rec is None:
                    break

                rec_data = bboxes = np.array(in_rec.getFirstLayerFp16()).reshape(30,1,37)
                decoded_text = codec.decode(rec_data)[0]
                texts.append(decoded_text)
                pos = rotated_rectangles[rec_received]

                if args.verbose:
                    print("{:2}: {:20}".format(rec_received, decoded_text),
                    "center({:3},{:3}) size({:3},{:3}) angle{:5.1f} deg".format(
                        int(pos[0][0]), int(pos[0][1]), pos[1][0], pos[1][1], pos[2]))
                    
                if args.preview == 2 and cropped_stacked is not None:
                    # Draw the text on the right side of 'cropped_stacked' - placeholder
                    cv2.putText(cropped_stacked, decoded_text,
                                    (120 + 10 , 32 * rec_received + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow('cropped_stacked', cropped_stacked)
                rec_received += 1

            if len(texts) > 0:
                print(texts)

            if cv2.waitKey(1) == ord('q'):
                break

            if rec_received >= rec_pushed:
                in_det = q_det.tryGet()

                if in_det is not None:
                    frame = host_sync.get_msg(in_det.getSequenceNum()).getCvFrame().copy()

                    scores, geom1, geom2 = to_tensor_result(in_det).values()
                    scores = np.reshape(scores, (1, 1, 64, 64))
                    geom1 = np.reshape(geom1, (1, 4, 64, 64))
                    geom2 = np.reshape(geom2, (1, 1, 64, 64))

                    bboxes, confs, angles = east.decode_predictions(scores, geom1, geom2)
                    boxes, angles = east.non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
                    rotated_rectangles = [
                        east.get_cv_rotated_rect(bbox, angle * -1)
                        for (bbox, angle) in zip(boxes, angles)
                    ]

                    rec_received = 0
                    rec_pushed = len(rotated_rectangles)

                    if args.verbose and rec_pushed:
                        print("====== Pushing for recognition, count:", rec_pushed)

                    cropped_stacked = None

                    for idx, rotated_rect in enumerate(rotated_rectangles):
                        # Detections are done on 256x256 frames, we are sending back 1024x1024
                        # That's why we multiply center and size values by 4
                        rotated_rect[0][0] = rotated_rect[0][0] * 4
                        rotated_rect[0][1] = rotated_rect[0][1] * 4
                        rotated_rect[1][0] = rotated_rect[1][0] * 4
                        rotated_rect[1][1] = rotated_rect[1][1] * 4

                        # Draw detection crop area on input frame
                        points = np.intp(cv2.boxPoints(rotated_rect))

                        if args.verbose:
                            print(rotated_rect)
                        
                        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

                        # TODO make it work taking args like in OpenCV:
                        # rr = ((256, 256), (128, 64), 30)
                        rr = dai.RotatedRect()
                        rr.center.x    = rotated_rect[0][0]
                        rr.center.y    = rotated_rect[0][1]
                        rr.size.width  = rotated_rect[1][0]
                        rr.size.height = rotated_rect[1][1]
                        rr.angle       = rotated_rect[2]
                        cfg = dai.ImageManipConfig()
                        cfg.setCropRotatedRect(rr, False)
                        cfg.setResize(120, 32)
                        # Send frame and config to device
                        if idx == 0:
                            w,h,c = frame.shape
                            imgFrame = dai.ImgFrame()
                            imgFrame.setData(to_planar(frame))
                            imgFrame.setType(dai.ImgFrame.Type.BGR888p)
                            imgFrame.setWidth(w)
                            imgFrame.setHeight(h)
                            q_manip_img.send(imgFrame)
                        else:
                            cfg.setReusePreviousImage(True)
                        q_manip_cfg.send(cfg)

                        # Get manipulated image from the device
                        transformed = q_manip_out.get().getCvFrame()

                        rec_placeholder_img = np.zeros((32, 200, 3), np.uint8)
                        transformed = np.hstack((transformed, rec_placeholder_img))

                        if args.preview == 2:
                            if cropped_stacked is None:
                                cropped_stacked = transformed
                            else:
                                cropped_stacked = np.vstack((cropped_stacked, transformed))

            if args.preview == 2 and cropped_stacked is not None:
                cv2.imshow('cropped_stacked', cropped_stacked)

            if args.preview >= 1 and frame is not None:
                cv2.imshow('frame', frame)

            key = cv2.waitKey(1)

            if  key == ord('q'):
                break
            # elif key == ord('t'):
            #     print("Autofocus trigger (and disable continuous)")
            #     ctrl = dai.CameraControl()
            #     ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            #     ctrl.setAutoFocusTrigger()
            #     controlQueue.send(ctrl)


if __name__ == '__main__':
    args = parse_args()
    main(args)
