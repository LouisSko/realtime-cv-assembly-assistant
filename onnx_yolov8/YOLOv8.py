import time
import cv2
import numpy as np
import onnxruntime

from onnx_yolov8.utils import xywh2xyxy, nms, draw_detections


class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.input_width = None
        self.input_height = None
        self.output_names = None
        self.input_shape = None
        self.input_names = None
        self.img_width = None
        self.img_height = None
        self.class_ids = []
        self.scores = []
        self.boxes = []
        self.comparison_frame_counter = 0
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.motion_prev = True

        # Settings
        self.multi_color = False
        self.displayAll = False
        self.displayConfidence = False
        self.displayLabel = True

        # Initialize model
        self.initialize_model(path)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def no_detections(self):
        self.boxes, self.scores, self.class_ids = [], [], []
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):

        # input image [1280, 720, 3]

        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor  # [1, 3, 640, 640]

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def set_settings(self, coloring, confidence, displayAll, displayConfidence, displayLabel):

        # Settings for displayed labels
        if confidence != "":
            # Convert to '%' if higher than 1
            if float(confidence) > 1:
                self.conf_threshold = float(confidence) / 100
            elif float(confidence) < 0.2:
                self.conf_threshold = 0.2
            else:
                self.conf_threshold = float(confidence)

        if coloring == "multi-color":
            self.multi_color = True
        if coloring == "single-color":
            self.multi_color = False

        self.displayAll = displayAll
        self.displayConfidence = displayConfidence
        self.displayLabel = displayLabel

    def draw_detections(self, image, required_class_ids=None, draw_scores=True, mask_alpha=0.2):

        return draw_detections(image, self.boxes, self.scores, self.class_ids, required_class_ids, mask_alpha,
                               self.multi_color, self.displayAll, self.displayConfidence, self.displayLabel)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
