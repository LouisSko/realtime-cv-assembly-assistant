import cv2
from flask import Flask, jsonify, render_template, Response, request
import requests

from onnx_yolov8.YOLOv8 import YOLOv8
from onnx_yolov8.utils import MotionDetector, gstreamer_pipeline


app = Flask(__name__, static_folder='resources')


LABELS = {
    0: "grey_beam_bent",
    1: "grey_axle_long_stop",
    2: "grey_axle_short_stop",
    3: "grey_axle_short",
    4: "grey_axle_long",
    5: "black_axle_pin_con",
    6: "black_beam",
    7: "black_pin_short",
    8: "red_oct_con",
    9: "red_pin_3L",
    10: "blue_pin_3L",
    11: "blue_axle_pin",
    12: "white_beam_bent",
    13: "white_beam_L",
    14: "white_tooth",
    15: "engine",
    16: "wheel",
    17: "wire"
}

STEPS_NO = {
    1: [4, 8, 15],
    2: [3],
    3: [0, 10],
    4: [10, 10, 13],
    5: [11, 11, 12],
    6: [1],
    7: [6],
    8: [7, 7],
    9: [16],
    10: [14],
    11: [2],
    12: [9, 9],
    13: [5, 7, 11, 11],
    14: [5, 7, 11, 11],
    15: [17]
}

STEPS = {
    1: [LABELS[8], LABELS[4], LABELS[15]],             # [red_oct_con, grey_axle_long, engine]
    2: [LABELS[3]],                                   # [grey_axle_short]
    3: [LABELS[0], LABELS[10]],                       # [grey_beam_bent, blue_pin_3L]
    4: [LABELS[10], LABELS[10], LABELS[13]],          # [blue_pin_3L, blue_pin_3L, white_beam_L]
    5: [LABELS[11], LABELS[11], LABELS[12]],          # [blue_axle_pin, blue_axle_pin, white_beam_bent]
    6: [LABELS[1]],                                   # [grey_axle_long_stop]
    7: [LABELS[6]],                                   # [black_beam]
    8: [LABELS[7], LABELS[7]],                        # [black_pin_short, black_pin_short]
    9: [LABELS[16]],                                  # [wheel]
    10: [LABELS[14]],                                 # [white_tooth]
    11: [LABELS[2]],                                  # [grey_axle_short_stop]
    12: [LABELS[9], LABELS[9]],                       # [red_pin_3L, red_pin_3L]
    13: [LABELS[11], LABELS[11], LABELS[5], LABELS[7]],# [blue_axle_pin, blue_axle_pin, black_axle_pin_con, black_pin_short]
    14: [LABELS[11], LABELS[11], LABELS[5], LABELS[7]],# [blue_axle_pin, blue_axle_pin, black_axle_pin_con, black_pin_short]
    15: [LABELS[17]]                                  # [wire]
}


# Define whether to use gstreamer pipeline or video
cap = cv2.VideoCapture('videos/IMG_4594.MOV')
#cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)


# Initialize YOLOv8 model
model_path = 'models/yolov8s_best.onnx'
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# th_diff=1 basically disables the detection
motion_detector = MotionDetector(threshold=20, th_diff=1, skip_frames=30)



def capture_camera():

    # initialise variables
    class_ids = []
    scores = []
    boxes = []
    motion = True

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
        #if cv2.waitKey(1) == ord('q'):
            #break

        pieces_url = 'http://127.0.0.1:5000/send-pieces'
        response = requests.get(pieces_url)
        if response.status_code == 200:
            pieces = response.json()  # is a list of labels e.g. ['grey4', 'wire']
        else:
            print('Error:', response.status_code)

        try:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                continue

            # check whether there is motion in the image
            motion = motion_detector.detect_motion(frame)
            #motion = False
            # Update object localizer if there is no motion in the image
            if not motion:
                boxes, scores, class_ids = yolov8_detector(frame, motion, skip_frames=0)

                frame = yolov8_detector.draw_detections(frame, required_class_ids=pieces)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(e)
            continue
        
    
        # Create a list to store the detection results
        detection_results = []
        # Format the detection results
        if len(boxes) > 0:
            for i in range(0, len(boxes)):
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
                
    # update motion information
    yolov8_detector.motion_prev = motion

    #cv2.imshow("Detected Objects", frame)


# Initialize global variables 
current_mode = 'Assembly'  # Default mode
current_step = 1  # Default step for assembly mode
detection_results = []
necessary_pieces = []


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

    return jsonify({'step': current_step,'pieces': STEPS[current_step], 'labels': STEPS_NO[current_step]})

# Go to previous instruction step
@app.route('/previous', methods=['POST'])
def previous_step():
    global current_step

    # Decrement the current step
    current_step -= 1

    # Check if we have reached the minimum step
    if current_step < 1:
        current_step = 1

    return jsonify({'step': current_step,'pieces': STEPS[current_step], 'labels': STEPS_NO[current_step]})


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
    if not isinstance(data, list) or not all(isinstance(d, dict) and 'label' in d and 'confidence' in d and 'boxes' in d for d in data):
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
                return jsonify({'message': 'There is {x} part missing. Check if all pieces are in the view of the camera.'.format(x=len(missing))})
            else:
                return jsonify({'message': 'There are {x} parts missing. Check if all pieces are in the view of the camera.'.format(x=len(missing))})
        else:
            return jsonify({'message': 'All necessary LEGO parts were found. Please grab the marked parts and follow the assembly instructions. Afterwards, press "Next steps" to continue.'})
        
    else:
        if len(labels) < len(STEPS_NO[current_step]):
            if len(missing) == 1:
                return jsonify({'message': 'You did not disassemble the correct parts. Make sure to only disassembly the parts displayed on the screen and place them within the view. There is {x} part missing.'.format(x=len(missing))})
            else:
                return jsonify({'message': 'You did not disassemble the correct parts. Make sure to only disassembly the parts displayed on the screen and place them within the view. There are {x} parts missing.'.format(x=len(missing))})
        else:
            return jsonify({'message': 'All necessary LEGO parts were disassembled correctly. Press "Next Step" to go to the next disassembly step.'})

# Handle error in case wrong page is opened
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404           

if __name__ == '__main__':
    app.run()
