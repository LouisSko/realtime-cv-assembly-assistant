import cv2
from flask import Flask, jsonify, render_template, Response, request

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

def get_instruction(detected_pieces):
    if STEPS[1] in detected_pieces:
        return 1
    elif STEPS[2] in detected_pieces:
        return 2
    elif STEPS[3] in detected_pieces:
        return 3
    elif STEPS[4] in detected_pieces:
        return 4
    elif STEPS[5] in detected_pieces:
        return 5
    elif STEPS[6] in detected_pieces:
        return 6
    elif STEPS[7] in detected_pieces:
        return 7
    elif STEPS[8] in detected_pieces:
        return 8
    elif STEPS[9] in detected_pieces:
        return 9
    elif STEPS[10] in detected_pieces:
        return 10
    elif STEPS[11] in detected_pieces:
        return 11
    elif STEPS[12] in detected_pieces:
        return 12
    elif STEPS[13] in detected_pieces:
        return 13
    elif STEPS[14] in detected_pieces:
        return 14
    elif STEPS[15] in detected_pieces:
        return 15
    else:
        return 0
    
current_step = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    instruction_image = '1.jpeg'
    return render_template('liveInstructions.html', title='LEGO Mindstorm: Real-time instruction manual', 
                           instruction_image=instruction_image, 
                           step=current_step, pieces=STEPS[current_step])

@app.route('/next', methods=['POST'])
def next_step():
    global current_step

    # Increment the current step
    current_step += 1

    # Check if we have reached the maximum step
    if current_step > 12:
        current_step = 12

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

@app.route('/video_feed')
def video_feed():
    return Response(capture_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live', methods=['POST'])
def detect():
    detected_pieces = request.json.get('detected_pieces')
    instruction_image = get_instruction(detected_pieces)
    return render_template('liveInstructions.html', title='LEGO Mindstorm: Real-time instruction manual', instruction_image=str(instruction_image)+".jpeg")

if __name__ == '__main__':
    app.run()
