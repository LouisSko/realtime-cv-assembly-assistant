# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
# from yolov8.YOLOv8_original import YOLOv8
from yolov8.YOLOV8_test_trt import YOLOv8

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""


def gstreamer_pipeline(
        sensor_id=0,
        capture_width=3246,
        capture_height=1848,
        display_width=1280,
        display_height=720,
        framerate=10,
        flip_method=0,
):
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


TRT_MODEL_PATH = 'yolov8n_best.trt'

# Load the TensorRT engine
with open(TRT_MODEL_PATH, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()


# For pre-processing the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))  # the input size of your YOLO model might be different
    image = (image / 255.0).astype(np.float32)
    return image


# Your function to process the raw output
def process_detections(detections):
    # Parse the raw output of YOLO model and return boxes, scores, class_ids
    pass

def detections():
    window_title = "CSI Camera"
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()

                if frame is None:  # timeout
                    continue

                # Preprocess the input frame
                frame = preprocess_image(frame)

                # Allocate memory for the input
                input_image_device = cuda.mem_alloc(1 * frame.nbytes)
                cuda.memcpy_htod(input_image_device, frame)

                # Allocate memory for the output
                output = np.empty(1000, dtype=np.float32)  # Adjust size according to your model
                output_device = cuda.mem_alloc(1 * output.nbytes)

                # Run the inference
                context.execute(bindings=[int(input_image_device), int(output_device)])

                # Copy the output from the device to the host
                cuda.memcpy_dtoh(output, output_device)

                # Process the detections
                boxes, scores, class_ids = process_detections(output)

                # Draw the detections on the original frame
                combined_img = draw_detections(frame, boxes, scores, class_ids)

                # If the window is still open, show the image
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, combined_img)
                else:
                    break

                # Stop the program on the ESC key or 'q'
                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


def draw_detections(image, boxes, scores, class_ids):
    # This function should draw the bounding boxes, class labels, and scores onto the image and return the combined_img
    pass


if __name__ == "__main__":
    detections()
