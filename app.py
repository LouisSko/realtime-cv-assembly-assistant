import cv2
from flask import Flask, jsonify, render_template, Response, request
import requests

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

# Function to capture the camera feed
def capture_camera():
    camera = cv2.VideoCapture(0)  # Use the appropriate camera index if necessary

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.release()
    
current_mode = 'assembly'  # Default mode
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
    print(settings)
    return jsonify('Success')

# Get user settings
@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)

# Set mode and first instruction
@app.route('/start', methods=['POST'])
def start():
    global current_mode, current_step

    current_mode = request.json['mode']
    if current_mode == 'assembly':
        current_step = 1
    elif current_mode == 'disassembly':
        current_step = 15

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})

# Load live instructions
@app.route('/live')
def live():
    global current_step
    instruction_image = 'resources/{}.jpeg'.format(current_step)
    return render_template('liveVideoInstruction.html', 
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
    print(necessary_pieces)

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
    # Get the detection results from the request
    detection_results = request.get_json()
    print(detection_results)

    # Return a response indicating successful handling of the detection results
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

    # Get all relevant detections
    for detection in detection_results:
        if int(detection['label']) in STEPS_NO[current_step]:
            labels.append(int(detection['label']))

    # Check if all necessary parts were detected
    for part in STEPS_NO[current_step]:
        if part not in labels:
            return jsonify({'message': 'Necessary pieces were not found. Check if all pieces are in the image.'})

    if current_mode == "Assembly":

        # Check if not enough parts were detected in case of two of same kind are needed
        if len(labels) < len(STEPS_NO[current_step]):
            return jsonify({'message': 'Necessary pieces were not found. Check if all pieces are in the image.'})
        else:
            return jsonify({'message': 'All necessary pieces were found. Grab the marked pieces from the video, which you can also see in the instruction picture, and follow the instructions.'})
        
    else:
        if len(labels) < len(STEPS_NO[current_step]):
            return jsonify({'message': 'You did not disassemble the correct parts. Make sure to only disassembly the parts displayed on the screen and place them within the image.'})
        else:
            return jsonify({'message': 'All necessary pieces were found. Press "Back" to go to the next disassembly step.'})
               
if __name__ == '__main__':
    app.run()
