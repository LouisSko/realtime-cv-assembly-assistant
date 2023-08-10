from flask import Flask, jsonify, render_template, request
from onnx_yolov8.utils import get_labels_steps

# Configurations for the assembly steps
# Get the labels and steps information
LABELS, STEPS_NO, STEPS = get_labels_steps()
# Set the default mode, step for assembly, detection results and necessary pieces
current_mode = 'Assembly'
current_step = 1
detection_results = []
necessary_pieces = []

# Start of Flask application definition
app = Flask(__name__, static_folder='resources')


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
    
@app.route('/end')
def end():
    return render_template('end.html')


# Go to next instruction step
@app.route('/next', methods=['POST'])
def next_step():
    global current_step

    # Increment the current step
    current_step += 1

    # Check if we have reached the maximum step
    if current_step > 15:
        return render_template('end.html')

    return jsonify({'step': current_step, 'pieces': STEPS[current_step], 'labels': STEPS_NO[current_step]})


# Go to previous instruction step
@app.route('/previous', methods=['POST'])
def previous_step():
    global current_step

    # Decrement the current step
    current_step -= 1

    # Check if we have reached the minimum step
    if current_step < 1:
        return render_template('end.html')

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

'''
@app.route('/video_feed')
def video_feed():
    return Response(capture_camera(model_path, video_source, use_camera_stream, skip_frames), mimetype='multipart/x-mixed-replace; boundary=frame')
'''

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
