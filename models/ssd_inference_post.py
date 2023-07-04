import sys
import argparse
import requests

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log, cudaDrawRect

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
#
net = detectNet(model="ssd_lego.onnx", labels="labels.txt",
                 input_blob="input_0", output_cvg="scores", output_bbox="boxes",
                 threshold=0.5)

# Initialize the video sources, outputs, and the object detection network

while True:
    # capture the next image
    img = input.Capture()

    if img is None:  # timeout
        continue

    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)
    cudaDrawRect(img, (200, 25, 350, 250), (255, 127, 0, 200))

    # print the detections
    #print("detected {:d} objects in image".format(len(detections)))

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    #TODO: Add functionality to only show overlays of sent pieces and then only post these
    pieces_url = 'http://127.0.0.1:5000/send-pieces'
    response = requests.get(pieces_url)
    if response.status_code == 200:
        pieces = response.json() # is a list of labels e.g. ['grey4', 'wire']

        print(pieces)
    else:
        print('Error:', response.status_code)

    # Create a list to store the detection results
    detection_results = []

    # Format the detection results
    for detection in detections:
        result = {
            'label': detection.ClassID,
            'confidence': detection.Confidence,
            'bounding_box': {
                'left': detection.Left,
                'top': detection.Top,
                'right': detection.Right,
                'bottom': detection.Bottom
            }
        }
        detection_results.append(result)

    if len(detection_results)>0:
        # Send the detection results to the web server
        url = 'http://127.0.0.1:5000/detections'  # Update with the appropriate URL
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=detection_results, headers=headers)

        # Check the response status code
        if response.status_code == 200:
            print("Detection results sent successfully")
        else:
            print("Error sending detection results")

        detections_url = 'http://127.0.0.1:5000/detections'
        response = requests.get(detections_url)
        if response.status_code == 200:
            detections = response.json()

            # Accessing objects in the JSON data
            for detection in detections:
                label = detection['label']
                confidence = detection['confidence']
                print(label)
                print(confidence)
        else:
            print('Error:', response.status_code)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
