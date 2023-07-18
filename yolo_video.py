import cv2
from onnx_yolov8.YOLOv8 import YOLOv8
from onnx_yolov8.utils import MotionDetector
import requests
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)
# Initialize YOLOv8 model
model_path = 'models/yolov8n_best.onnx'
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Initialize video
cap = cv2.VideoCapture('videos/IMG_4594.MOV')

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# th_diff=1 basically diables the detection
motion_detector = MotionDetector(threshold=20, th_diff=1, skip_frames=30)

while cap.isOpened():

    settings_url = 'http://127.0.0.1:5000/settings'
    response_settings = requests.get(settings_url)
    if response_settings.status_code == 200:
        settings_resp = response_settings.json()
        coloring = settings_resp['coloring']
        confidence = settings_resp['confidence']
        displayConfidence = settings_resp['displayConfidence']
        displayLabel = settings_resp['displayLabel']
        displayAll = settings_resp['displayAll']

        yolov8_detector.set_settings(coloring, confidence, displayAll, displayConfidence, displayLabel)

    else:
        print('Error:', response_settings.status_code)

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            continue

        # check whether there is motion in the image
        motion = motion_detector.detect_motion(frame)
        motion = False
        # Update object localizer if there is no motion in the image
        if not motion:
            boxes, scores, class_ids = yolov8_detector(frame, motion, skip_frames=0)
            frame = yolov8_detector.draw_detections(frame, required_class_ids=["red1", "grey5", "engine"])

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        
    except Exception as e:
        print(e)
        continue

    # Create a list to store the detection results
    detection_results = []
    # Format the detection results
    if len(boxes)>0:
        for i in range(0,len(boxes)):
            result = {
                'label': str(class_ids[i]),
                'confidence': str(scores[i]),
                'boxes': str(boxes[i])
            }
            detection_results.append(result)

        url = 'http://127.0.0.1:5000/detections'  # Update with the appropriate URL
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=detection_results, headers=headers)

        # Check the response status code
        if response.status_code != 200:
            print("Error sending detection results")


    
    # check whether there is motion in the image
    motion = motion_detector.detect_motion(frame)
    '''
    # Update object localizer if there is no motion in the image
    if not motion:
        boxes, scores, class_ids = yolov8_detector(frame, motion, skip_frames=0)

        frame = yolov8_detector.draw_detections(frame, required_class_ids = pieces)

        # Create a list to store the detection results
        detection_results = []

        # Format the detection results
        if len(boxes)>0:
            for i in range(0,len(boxes)):
                result = {
                    'label': str(class_ids[i]),
                    'confidence': str(scores[i]),
                    'boxes': str(boxes[i])
                }
                detection_results.append(result)

            url = 'http://127.0.0.1:5000/detections'  # Update with the appropriate URL
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=detection_results, headers=headers)

            # Check the response status code
            if response.status_code == 200:
                print("Detection results sent successfully")
            else:
                print("Error sending detection results")
    '''
    yolov8_detector.motion_prev = motion
    
    cv2.imshow("Detected Objects", frame)

    #out.release()
