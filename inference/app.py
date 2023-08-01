import os
import sys

# Go up one directory from the current script's directory -> necessary for imports
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_PATH)

import cv2
from flask import Flask, jsonify, render_template, Response, request
import requests
from onnx_yolov8.YOLOv8 import YOLOv8
from onnx_yolov8.utils import MotionDetector, get_labels_steps

# Determine the path to the file dynamically, based on the location of the currently-running script: -> necessary for loading model
current_dir = os.path.dirname(os.path.realpath(__file__))


# Start of Flask application definition
app = Flask(__name__, static_folder='resources')

# Configurations for the video capture method
# Define whether to use gstreamer pipeline or video
video_path = os.path.join(current_dir, '../videos/IMG_4594.MOV')
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER) # Uncomment line to use gstreamer pipeline

# Configurations for the YOLOv8 model
# Initialize the YOLOv8 model with given configurations
model_path = os.path.join(current_dir, '../models/yolov8s_best.onnx')
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Configurations for the Motion Detector
# Initialize the MotionDetector with given configurations
motion_detector = MotionDetector(threshold=20, th_diff=1, skip_frames=30)

# Configurations for the assembly steps
# Get the labels and steps information
LABELS, STEPS_NO, STEPS = get_labels_steps()
# Set the default mode, step for assembly, detection results and necessary pieces
current_mode = 'Assembly'
current_step = 1
detection_results = []
necessary_pieces = []


def capture_camera():
    """
    This function is responsible for capturing frames from the camera,
    performing object detection, and returning the detection results.
    """

    # Initialize variables
    frame_counter = 0
    skip_frames = 5  # number of frames to skip before performing detection

    # Define endpoints for settings, pieces, and detections
    settings_endpoint = 'http://127.0.0.1:5000/settings'
    pieces_endpoint = 'http://127.0.0.1:5000/send-pieces'
    detections_endpoint = 'http://127.0.0.1:5000/detections'

    while cap.isOpened():

        # Read frame from the video
        retrieved_frame, frame = cap.read()
        frame_counter += 1

        # Continue to the next iteration if no frame is retrieved
        if not retrieved_frame:
            continue

        # Perform detection every n frames (determined by `skip_frames`)
        if frame_counter >= skip_frames:

            # Reset frame counter
            frame_counter = 0

            # Update detector settings and get required pieces
            update_detector_settings(yolov8_detector, settings_endpoint)
            pieces = get_required_pieces(pieces_endpoint)

            # Detect motion in the frame
            motion = motion_detector.detect_motion(frame)

            # If there is motion, no detections are performed
            if motion:
                boxes, scores, class_ids = yolov8_detector.no_detections()
            else:  # If there is no motion, perform detections
                boxes, scores, class_ids = yolov8_detector.detect_objects(frame)

            # Format and post detection results
            detection_results = format_detection_results(boxes, class_ids, scores)
            post_detection_results(detection_results, detections_endpoint)

            # Draw detections on the frame and convert it to jpeg format
            frame = yolov8_detector.draw_detections(frame, required_class_ids=pieces)
            jpeg_frame = encode_frame_to_jpeg(frame)

            # Yield the frame in HTTP response format
            yield format_http_response(jpeg_frame)


def encode_frame_to_jpeg(frame):
    """
    This function encodes a given frame into JPEG format.

    :param frame: The frame to be encoded.
    :return: The frame in JPEG format.
    """
    _, buffer = cv2.imencode('.jpg', frame)
    jpeg_frame = buffer.tobytes()
    return jpeg_frame


def format_http_response(jpeg_frame):
    """
    This function formats the given JPEG frame into a HTTP response.

    :param jpeg_frame: The JPEG frame to be sent in the response.
    :return: The HTTP response.
    """
    http_response = (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n\r\n')
    return http_response


def get_response_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print('Error:', response.status_code)
        return None


def update_detector_settings(detector, settings_url):
    settings_resp = get_response_from_url(settings_url)
    if settings_resp is not None:
        coloring = settings_resp['coloring']
        confidence = settings_resp['confidence']
        displayConfidence = settings_resp['displayConfidence']
        displayLabel = settings_resp['displayLabel']
        displayAll = settings_resp['displayAll']

        detector.set_settings(coloring, confidence, displayAll, displayConfidence, displayLabel)


def get_required_pieces(pieces_url):
    response = get_response_from_url(pieces_url)
    if response is not None:
        return response  # is a list of labels e.g. ['grey4', 'wire']
    return None


def format_detection_results(boxes, class_ids, scores):
    detection_results = [{'label': str(class_id), 'confidence': str(score), 'boxes': str(box)}
                         for box, class_id, score in zip(boxes, class_ids, scores)]

    return detection_results


def post_detection_results(detection_results, detection_url):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(detection_url, json=detection_results, headers=headers)

    # Check the response status code
    if response.status_code != 200:
        print("Error sending detection results")


# Load Homepage
@app.route('/')
def index():
    return render_template('index.html')


# Post user settings
@app.route('/settings', methods=['POST'])
def set_settings():
    global settings
    settings = request.get_json()
    return jsonify('Success')


# Get user settings
@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)


# Set mode and first instruction
@app.route('/start', methods=['POST'])
def start():
    global current_mode, current_step

    data = request.get_json()
    mode = data.get('mode')

    if mode not in ['Assembly', 'Disassembly']:
        return jsonify({'error': 'Invalid mode'}), 400

    current_mode = mode
    current_step = 1 if mode == 'Assembly' else 15

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})


# Load live instructions
@app.route('/live')
def live():
    global current_step, current_mode
    instruction_image = 'resources/{}.jpeg'.format(current_step)
    if current_mode == 'Assembly':
        return render_template('liveInstructionsAssembly.html',
                               instruction_image=instruction_image,
                               step=current_step, pieces=STEPS[current_step])
    else:
        return render_template('liveInstructionsDisassembly.html',
                               instruction_image=instruction_image,
                               step=current_step, pieces=STEPS[current_step])


# Go to next instruction step
@app.route('/next', methods=['POST'])
def next_step():
    global current_step

    # Increment the current step
    current_step += 1

    # Check if we have reached the maximum step
    if current_step > 15:
        current_step = 15

    return jsonify({'step': current_step, 'pieces': STEPS[current_step], 'labels': STEPS_NO[current_step]})


# Go to previous instruction step
@app.route('/previous', methods=['POST'])
def previous_step():
    global current_step

    # Decrement the current step
    current_step -= 1

    # Check if we have reached the minimum step
    if current_step < 1:
        current_step = 1

    return jsonify({'step': current_step, 'pieces': STEPS[current_step], 'labels': STEPS_NO[current_step]})


# POST all necessary pieces of current instruction step
@app.route('/send-pieces', methods=['POST'])
def send_pieces():
    global necessary_pieces
    # Get the necessary pieces from the request payload
    necessary_pieces = request.json['pieces']

    # Return a response to indicate successful processing
    return jsonify({'message': 'Necessary pieces sent successfully'})


# GET all necssary pices of current instruction stepp
@app.route('/send-pieces', methods=['GET'])
def get_pieces():
    return jsonify(necessary_pieces)


@app.route('/video_feed')
def video_feed():
    return Response(capture_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


# POST detected pieces
@app.route('/detections', methods=['POST'])
def handle_detections():
    global detection_results

    data = request.get_json()
    if not isinstance(data, list) or not all(
            isinstance(d, dict) and 'label' in d and 'confidence' in d and 'boxes' in d for d in data):
        return 'Invalid detection results data', 400

    detection_results = data

    return 'Detection results received.'


# GET detected pieces
@app.route('/detections', methods=['GET'])
def get_detections():
    return jsonify(detection_results)


# POST results of check
@app.route('/labels', methods=['POST'])
def handle_labels():
    global current_step, current_mode, detection_results

    labels = []
    missing = []

    # Get all relevant detections
    for detection in detection_results:
        if int(detection['label']) in STEPS_NO[current_step]:
            labels.append(int(detection['label']))

    # Check if all necessary parts were detected
    for part in STEPS_NO[current_step]:
        if part not in labels:
            missing.append(part)

    if current_mode == "Assembly":

        # Check if not enough parts were detected in case of two of same kind are needed
        if len(labels) < len(STEPS_NO[current_step]):
            if len(missing) == 1:
                return jsonify({
                    'message': 'There is {x} part missing. Check if all pieces are in the view of the camera.'.format(
                        x=len(missing))})
            else:
                return jsonify({
                    'message': 'There are {x} parts missing. Check if all pieces are in the view of the camera.'.format(
                        x=len(missing))})
        else:
            return jsonify({
                'message': 'All necessary LEGO parts were found. Please grab the marked parts and follow the assembly instructions. Afterwards, press "Next steps" to continue.'})

    else:
        if len(labels) < len(STEPS_NO[current_step]):
            if len(missing) == 1:
                return jsonify({
                    'message': 'You did not disassemble the correct parts. Make sure to only disassembly the parts displayed on the screen and place them within the view. There is {x} part missing.'.format(
                        x=len(missing))})
            else:
                return jsonify({
                    'message': 'You did not disassemble the correct parts. Make sure to only disassembly the parts displayed on the screen and place them within the view. There are {x} parts missing.'.format(
                        x=len(missing))})
        else:
            return jsonify({
                'message': 'All necessary LEGO parts were disassembled correctly. Press "Next Step" to go to the next disassembly step.'})


# Handle error in case wrong page is opened
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run()
