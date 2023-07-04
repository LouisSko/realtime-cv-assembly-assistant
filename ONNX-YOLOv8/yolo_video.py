import cv2
from yolov8 import YOLOv8
import cv2

import cv2
from yolov8.utils import MotionDetector


# Initialize YOLOv8 model
model_path = '../models/yolov8n_best.onnx'
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Initialize video
cap = cv2.VideoCapture('../videos/IMG_4594.MOV')

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# th_diff=1 basically diables the detection
motion_detector = MotionDetector(threshold=20, th_diff=1, skip_frames=30)

while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # check whether there is motion in the image
    motion = motion_detector.detect_motion(frame)

    # Update object localizer if there is no motion in the image
    if not motion:
        boxes, scores, class_ids = yolov8_detector(frame, motion, skip_frames=0)

        frame = yolov8_detector.draw_detections(frame)

    yolov8_detector.motion_prev = motion

    cv2.imshow("Detected Objects", frame)

# out.release()
