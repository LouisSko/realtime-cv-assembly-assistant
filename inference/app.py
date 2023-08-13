import os
import sys

# Go up one directory from the current script's directory -> necessary for imports
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_PATH)

# Determine the path to the file dynamically, based on the location of the currently-running script: -> necessary for loading model
current_dir = os.path.dirname(os.path.realpath(__file__))

import cv2
from flask import Flask, jsonify, render_template, Response, request
import requests
from onnx_yolov8.YOLOv8 import YOLOv8
from onnx_yolov8.utils import MotionDetector, get_labels_steps, gstreamer_pipeline
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Process a video file or gstreamer pipeline with a specified model.')
parser.add_argument('--model_path', type=str, default='../models/yolov8s_best.onnx',
                    help='Path to the model file.')
parser.add_argument('--video_source', type=str, default='../videos/blue_pin_long.mp4',
                    help='Path to the video file to be processed.')
parser.add_argument('--use_camera_stream', action='store_true',  # if the command line argument is specified, then argparse will assign the value True
                    help='Use this flag if the camera stream of the nano should be used.')
parser.add_argument('--skip_frames', type=int, default=5,
                    help='Make detections and send information only every n frames')

args = parser.parse_args()

# assign command line arguments to global variables
model_path = args.model_path
video_source = args.video_source
use_camera_stream = args.use_camera_stream
skip_frames = args.skip_frames


# Configurations for the assembly steps
# Get the labels and steps information
LABELS, STEPS_NO, STEPS = get_labels_steps()
# Set the default mode, step for assembly, detection results and necessary pieces
current_mode = 'Assembly'
current_step = 1
detection_results = []
necessary_pieces = []


def capture_camera(model_path, video_source, use_camera_stream, skip_frames):
    """
    Capture video frames from a camera or video source and perform object detection.

    This function captures video frames, detects objects using a YOLOv8 model, and
    returns the detection results. It also integrates a motion detection mechanism
    to optimize the detection process.

    Parameters:
    - model_path (str): Path to the YOLOv8 model weights.
    - video_source (str): Path to the video file or camera source identifier.
    - use_camera_stream (bool): Whether to capture from a camera stream using gstreamer.
    - skip_frames (int): Number of frames to skip before performing detection.

    Yields:
    - bytes: A JPEG-encoded frame with drawn detections in HTTP response format.
    """

    if use_camera_stream:
        # Configurations for the video capture method using gstreamer pipeline
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    else:
        # Configurations for the video capture method using a video file
        video_path = os.path.join(current_dir, video_source)
        cap = cv2.VideoCapture(video_path)

    # Initialize the YOLOv8 model with given configurations
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

    # Configurations for the Motion Detector
    # Initialize the MotionDetector with given configurations
    motion_detector = MotionDetector(threshold=20, th_diff=1, skip_frames=30)

    # Initialize variables
    frame_counter = 0
    # skip_frames number of frames to skip before performing detection

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

            # If motion is detected, skip detections; otherwise, perform object detection
            if motion:
                boxes, scores, class_ids = yolov8_detector.no_detections()
            else:
                boxes, scores, class_ids = yolov8_detector.detect_objects(frame)

            # Format and post detection results
            detection_results = format_detection_results(boxes, class_ids, scores)
            post_detection_results(detection_results, detections_endpoint)

            # Draw detections on the frame and convert it to jpeg format
            frame = yolov8_detector.draw_detections(frame, required_class_ids=pieces)
            jpeg_frame = encode_frame_to_jpeg(frame)

            # Return the JPEG frame in HTTP response format
            yield format_http_response(jpeg_frame)


def encode_frame_to_jpeg(frame):
    """
    This function encodes a given frame into JPEG format.

    :param frame: The frame to be encoded.
    :return: The frame in JPEG format.
    """

    # Use OpenCV's imencode function to encode the frame into JPEG format
    # The function returns two values:
    # 1. A boolean indicating the success of the operation
    # 2. The encoded image as an array of bytes (buffer)
    _, buffer = cv2.imencode('.jpg', frame)

    # Convert the buffer array (encoded image) into a flat byte string
    jpeg_frame = buffer.tobytes()

    # Return the encoded frame in JPEG format as a bytes object
    return jpeg_frame


def format_http_response(jpeg_frame):
    """
    This function formats the given JPEG frame into a HTTP response suitable for MJPEG streaming.

    :param jpeg_frame: The JPEG frame to be sent in the response.
    :return: The HTTP response.
    """

    # The response is constructed for MJPEG streaming.
    # It starts with the boundary string '--frame' followed by a CRLF (\r\n).
    # Then, the Content-Type is set to 'image/jpeg' indicating the type of the content.
    # Two CRLFs (\r\n\r\n) are used to separate the headers from the body.
    # The body contains the actual JPEG frame.
    # Finally, two more CRLFs denote the end of this part of the response.
    http_response = (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n\r\n')

    # Return the formatted HTTP response
    return http_response


def get_response_from_url(url):
    """
    Get a JSON response from a given URL.

    :param url (str): The URL to fetch the JSON response from.
    :return: dict or None: The JSON response if successful, None if not.
    """

    # Make a GET request to the provided URL
    response = requests.get(url)

    # Check if the response status code is 200 (HTTP OK)
    # If yes, parse the response as JSON and return it
    if response.status_code == 200:
        return response.json()

    # If the response status code is not 200, print an error message
    # and return None
    else:
        print('Error:', response.status_code)
        return None


def update_detector_settings(detector, settings_url):
    """
    Update the settings of a detector based on the provided URL.
    
    param: 
        detector: The detector object to update settings for.
        settings_url (str): The URL to fetch detector settings from.
    """
    settings_resp = get_response_from_url(settings_url)
    if settings_resp is not None:
        coloring = settings_resp['coloring']
        confidence = settings_resp['confidence']
        displayConfidence = settings_resp['displayConfidence']
        displayLabel = settings_resp['displayLabel']
        displayAll = settings_resp['displayAll']

        detector.set_settings(coloring, confidence, displayAll, displayConfidence, displayLabel)


def get_required_pieces(pieces_url):
    """
    Get the required pieces from the provided URL.

    param: pieces_url (str): The URL to fetch the required pieces from.

    return: List of labels representing the required pieces.
    """
    response = get_response_from_url(pieces_url)
    if response is not None:
        return response  # is a list of labels e.g. ['grey4', 'wire']
    return None


def format_detection_results(boxes, class_ids, scores):
    """
    Format detection results into a consistent structure.

    param:
        boxes: List of bounding boxes for detected objects.
        class_ids: List of class IDs for detected objects.
        scores: List of confidence scores for detected objects.

    return: List of dictionaries with formatted detection results.
    """
    detection_results = [{'label': str(class_id), 'confidence': str(score), 'boxes': str(box)}
                         for box, class_id, score in zip(boxes, class_ids, scores)]

    return detection_results


def post_detection_results(detection_results, detection_url):
    """
    Post detection results to the specified URL.

    param:
        detection_results: List of dictionaries containing detection results.
        detection_url (str): The URL to send the detection results to.
    """
    headers = {'Content-Type': 'application/json'}
    response = requests.post(detection_url, json=detection_results, headers=headers)

    # Check the response status code
    if response.status_code != 200:
        print("Error sending detection results")


# Start of Flask application definition
app = Flask(__name__, static_folder='resources')


@app.route('/')
def index():
    """
    Load the homepage.
    
    Returns:
        Rendered template for the homepage.
    """
    return render_template('index.html')


@app.route('/settings', methods=['POST'])
def set_settings():
    """
    Post user settings.
    
    Expects JSON data with user settings and updates the global settings variable.
    
    Returns:
        JSON response confirming the success of the settings update.
    """
    global settings
    settings = request.get_json()
    return jsonify('Success')


@app.route('/settings', methods=['GET'])
def get_settings():
    """
    Get user settings.
    
    Returns:
        JSON response with the current user settings.
    """
    return jsonify(settings)


@app.route('/start', methods=['POST'])
def start():
    """
    Set the mode and first instruction step.
    
    Expects JSON data with 'mode' field ('Assembly' or 'Disassembly').
    Sets the current_mode and current_step accordingly.
    
    Returns:
        JSON response with 'step' and 'pieces' for the initial instruction step.
    """
    global current_mode, current_step

    data = request.get_json()
    mode = data.get('mode')

    if mode not in ['Assembly', 'Disassembly']:
        return jsonify({'error': 'Invalid mode'}), 400

    current_mode = mode
    current_step = 1 if mode == 'Assembly' else 15

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})


@app.route('/live')
def live():
    """
    Load live instruction page based on current mode and step.
    
    Renders live instruction template with appropriate data.
    
    Returns:
        Rendered template for live instruction.
    """
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
    
    
@app.route('/end')
def end():
    """
    Load end page.
    
    Returns:
        Rendered template for the end page.
    """
    return render_template('end.html')


@app.route('/next', methods=['POST'])
def next_step():
    """
    Move to the next instruction step.
    
    Increments the current_step. If it's the last step, redirects to the end page.
    
    Returns:
        JSON response with 'step', 'pieces', and 'labels' for the next step.
    """
    global current_step

    # Increment the current step
    current_step += 1

    # Check if we have reached the maximum step
    if current_step > 15:
        return render_template('end.html')

    return jsonify({'step': current_step, 'pieces': STEPS[current_step], 'labels': STEPS_NO[current_step]})


@app.route('/previous', methods=['POST'])
def previous_step():
    """
    Move to the previous instruction step.
    
    Decrements the current_step. If it's the first step, redirects to the end page.
    
    Returns:
        JSON response with 'step', 'pieces', and 'labels' for the previous step.
    """
    global current_step

    # Decrement the current step
    current_step -= 1

    # Check if we have reached the minimum step
    if current_step < 1:
        return render_template('end.html')

    return jsonify({'step': current_step, 'pieces': STEPS[current_step], 'labels': STEPS_NO[current_step]})


@app.route('/send-pieces', methods=['POST'])
def send_pieces():
    """
    Store the necessary pieces for the current instruction step.
    
    Expects JSON data with 'pieces' field containing a list of necessary pieces.
    
    Returns:
        JSON response with success message.
    """
    global necessary_pieces
    # Get the necessary pieces from the request payload
    necessary_pieces = request.json['pieces']

    # Return a response to indicate successful processing
    return jsonify({'message': 'Necessary pieces sent successfully'})


@app.route('/send-pieces', methods=['GET'])
def get_pieces():
    """
    Get the necessary pieces for the current instruction step.
    
    Returns:
        JSON response with the list of necessary pieces.
    """
    return jsonify(necessary_pieces)


@app.route('/video_feed')
def video_feed():
    """
    Stream video feed from the camera with detected objects overlay.
    
    Returns:
        Response containing the video stream with detected objects overlay.
    """
    return Response(capture_camera(model_path, video_source, use_camera_stream, skip_frames), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections', methods=['POST'])
def handle_detections():
    """
    Handle the detection results sent by the client.
    
    Expects JSON data with a list of dictionaries containing 'label', 'confidence', and 'boxes'.
    Checks the validity of the data and stores the detection results.
    
    Returns:
        Response confirming the receipt of detection results or an error message.
    """
    global detection_results

    data = request.get_json()
    if not isinstance(data, list) or not all(
            isinstance(d, dict) and 'label' in d and 'confidence' in d and 'boxes' in d for d in data):
        return 'Invalid detection results data', 400

    detection_results = data

    return 'Detection results received.'


@app.route('/detections', methods=['GET'])
def get_detections():
    """
    Get the stored detection results.
    
    Returns:
        JSON response with the stored detection results.
    """
    return jsonify(detection_results)


@app.route('/labels', methods=['POST'])
def handle_labels():
    """
    Handle the results of checking detected pieces against expected pieces.
    
    Checks if the necessary pieces for the current instruction step are correctly detected.
    Returns a message indicating the correctness of the detection.
    
    Returns:
        JSON response with a message about the correctness of detected pieces.
    """
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


@app.errorhandler(404)
def page_not_found(error):
    """
    Handle error when a wrong page is opened.
    
    Returns:
        Rendered template for a 404 error page.
    """
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run()