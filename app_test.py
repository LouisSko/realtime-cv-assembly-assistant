import cv2
from flask import Flask, jsonify, render_template, Response, request
import requests

from onnx_yolov8.yolov8.YOLOv8 import YOLOv8
from onnx_yolov8.yolov8.utils import MotionDetector

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

# Initialize YOLOv8 model
model_path = 'models/yolov8n_best.onnx'
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Initialize video
cap = cv2.VideoCapture('videos/IMG_4594.MOV')

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# th_diff=1 basically diables the detection
motion_detector = MotionDetector(threshold=20, th_diff=1, skip_frames=30)

def capture_camera():
    while cap.isOpened():

        # Press key q to stop
        #if cv2.waitKey(1) == ord('q'):
            #break

        try:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(e)
            continue

        #TODO: Add functionality to only show overlays of sent pieces and then only post these
        pieces_url = 'http://127.0.0.1:5000/send-pieces'
        response = requests.get(pieces_url)
        if response.status_code == 200:
            pieces = response.json()  # is a list of labels e.g. ['grey4', 'wire']

            print(pieces)
        else:
            print('Error:', response.status_code)

        # check whether there is motion in the image
        '''motion = motion_detector.detect_motion(frame)

        # Update object localizer if there is no motion in the image
        if not motion:
            boxes, scores, class_ids = yolov8_detector(frame, motion, skip_frames=0)

            frame = yolov8_detector.draw_detections(frame)

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

        yolov8_detector.motion_prev = motion'''

        #cv2.imshow("Detected Objects", frame)

# Function to capture the camera feed
'''
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
'''
    
current_mode = 'assembly'  # Default mode
current_step = 1  # Default step for assembly mode
detection_results = []
necessary_pieces = []

# Load Homepage
@app.route('/')
def index():
    return render_template('index.html')

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
    return render_template('test.html', 
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

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})

# Go to previous instruction step
@app.route('/previous', methods=['POST'])
def previous_step():
    global current_step

    # Decrement the current step
    current_step -= 1

    # Check if we have reached the minimum step
    if current_step < 1:
        current_step = 1

    return jsonify({'step': current_step, 'pieces': STEPS[current_step]})

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
