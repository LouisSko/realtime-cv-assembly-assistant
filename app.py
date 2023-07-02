import cv2
from flask import Flask, jsonify, render_template, Response, request
import requests

app = Flask(__name__, static_folder='resources')

LABELS = {
    0: "grey1",
    1: "grey2",
    2: "grey3",
    3: "grey4",
    4: "grey5",
    5: "black1",
    6: "black2",
    7: "black3",
    8: "red1",
    9: "red2",
    10: "blue1",
    11: "blue2",
    12: "white1",
    13: "white2",
    14: "white3",
    15: "engine",
    16: "wheel",
    17: "wire"
}

STEPS = {
    1: ["red1", "grey5", "engine"],
    2: ["grey4"],
    3: ["grey1", "blue1"],
    4: ["blue1", "blue1", "white2"],
    5: ["blue2", "blue2", "white1"],
    6: ["grey2"],
    7: ["black2"],
    8: ["black3", "black3"],
    9: ["wheel"],
    10: ["white3"],
    11: ["grey3"],
    12: ["red2", "red2"],
    13: ["blue2", "blue2", "black1", "black3"],
    14: ["blue2", "blue2", "black1", "black3"],
    15: ["wire"]
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global current_mode, current_step

    current_mode = request.json['mode']
    if current_mode == 'assembly':
        current_step = 1
    elif current_mode == 'disassembly':
        current_step = 15

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})


@app.route('/live')
def live():
    global current_step
    instruction_image = 'resources/{}.jpeg'.format(current_step)
    return render_template('liveInstructions.html', 
                           instruction_image=instruction_image, 
                           step=current_step, pieces=STEPS[current_step])

@app.route('/next', methods=['POST'])
def next_step():
    global current_step

    # Increment the current step
    current_step += 1

    # Check if we have reached the maximum step
    if current_step > 15:
        current_step = 15

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})

@app.route('/previous', methods=['POST'])
def previous_step():
    global current_step

    # Decrement the current step
    current_step -= 1

    # Check if we have reached the minimum step
    if current_step < 1:
        current_step = 1

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})

@app.route('/send-pieces', methods=['POST'])
def send_pieces():
    # Get the necessary pieces from the request payload
    necessary_pieces = request.json['pieces']
    print(necessary_pieces)

    # TODO: Implement the logic to send the necessary pieces to the Jetson Nano

    # Return a response to indicate successful processing
    return jsonify({'message': 'Necessary pieces sent successfully'})

'''
@app.route('/video_feed')
def video_feed():
    return Response(capture_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
'''

@app.route('/detections', methods=['POST'])
def handle_detections():
    global detection_results
    # Get the detection results from the request
    detection_results = request.get_json()
    print(detection_results)

    # Process the detection results
    for detection in detection_results:
        label = detection['label']
        confidence = detection['confidence']
        bounding_box = detection['bounding_box']
    
        # Perform further processing or storage of the detection results here

    # Return a response indicating successful handling of the detection results
    return 'Detection results received and processed'

@app.route('/detections', methods=['GET'])
def get_detections():
    return jsonify(detection_results)

@app.route('/labels', methods=['POST'])
def handle_labels():
    global current_step, current_mode, detection_results

    print(detection_results)
    missing_labels = []

    for label in STEPS[current_step]:
            if label not in detection_results:
                missing_labels.append(label)

    if current_mode == 'assembly':

        if len(missing_labels)>0:
            return jsonify({'message': 'Necessary pieces were not found. Check if all pieces are in the image.', 'missing': missing_labels})
        else:
            return jsonify({'message': 'All necessary pieces were found. Select the marked pieces from the video, which you can also see in the instruction picture, and follow the instructions.', 'missing': missing_labels})
        
    else:
        if len(missing_labels)>0:
            return jsonify({'message': 'You did not disassemble the correct parts. Make sure to only disassembly the parts displayed on the screen and place them within the image.', 'missing': missing_labels})
        else:
            return jsonify({'message': 'All necessary pieces were found. Press "Back" to go to the next disassembly step.', 'missing': missing_labels})
        

if __name__ == '__main__':
    app.run()
