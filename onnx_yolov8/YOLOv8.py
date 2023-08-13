import cv2
import numpy as np
import onnxruntime

from onnx_yolov8.utils import xywh2xyxy, nms, draw_detections


class YOLOv8:
    """
    YOLOv8 (You Only Look Once) Object Detection model handler.

    This class provides an interface to interact with the YOLOv8 object detection model.
    It allows users to load a model, set various configurations, and perform object detection
    on images.

    Attributes:
    - conf_threshold (float): Confidence threshold for filtering weak detections.
    - iou_threshold (float): Intersection over Union threshold for non-maximum suppression.
    - multi_color (bool): Flag to determine if multi-color display is enabled.
    - displayAll (bool): Flag to determine if all detected objects should be displayed.
    - displayConfidence (bool): Flag to display confidence scores along with detections.
    - displayLabel (bool): Flag to display class labels along with detections.

    Example Usage:
    detector = YOLOv8(path='model_path.onnx')
    detections = detector.detect_objects(image)
    """

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        """
        Initialize the YOLOv8 object detector with configurations.

        Parameters:
        - path (str): Path to the pre-trained YOLOv8 model.
        - conf_thres (float, optional): Confidence threshold. Defaults to 0.7.
        - iou_thres (float, optional): IoU threshold for non-max suppression. Defaults to 0.5.
        """

        # Attributes for object detection
        self.session = None
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
        self.multi_color = False
        self.displayAll = False
        self.displayConfidence = False
        self.displayLabel = True

        # Initialize model
        self.initialize_model(path)

    def initialize_model(self, path):
        """
        Initialize the YOLOv8 object detection model.

        This method loads the pre-trained YOLOv8 model using ONNX runtime,
        and initializes the session for inference. It also retrieves
        essential details about the model's input and output layers.

        Args:
            path (str): Path to the pre-trained YOLOv8 ONNX model.
        """

        # Load the ONNX model using ONNX runtime with CUDA and CPU execution providers
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])

        # Extract model's input details (e.g., input shape, names)
        self.get_input_details()

        # Extract model's output details (e.g., output shape, names)
        self.get_output_details()

    def detect_objects(self, image):
        """
        Detect objects in the provided image using the YOLOv8 model.

        This method takes an image, prepares it for inference, runs the model
        to detect objects, and then processes the model's output to retrieve
        bounding boxes, scores, and class IDs of detected objects.

        Parameters:
        - image (numpy.ndarray): The input image in which to detect objects.

        Returns:
        - list: Detected bounding boxes.
        - list: Confidence scores corresponding to the detected bounding boxes.
        - list: Class IDs corresponding to the detected bounding boxes.
        """

        # Prepare the image for model input (resize, normalize, etc.)
        input_tensor = self.prepare_input(image)

        # Run the YOLOv8 model for object detection
        outputs = self.inference(input_tensor)

        # Process the raw model output to retrieve bounding boxes, scores, and class IDs
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def no_detections(self):
        """
        Reset the detection attributes of the YOLOv8 instance.

        This method clears the stored bounding boxes, scores, and class IDs,
        effectively indicating that no objects have been detected. It is used when
        detections from previous frames need to be cleared.

        Returns:
        - list: Empty list indicating no detected bounding boxes.
        - list: Empty list indicating no confidence scores for bounding boxes.
        - list: Empty list indicating no class IDs for bounding boxes.
        """

        # Resetting detection attributes to empty lists
        self.boxes, self.scores, self.class_ids = [], [], []

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        """
        Prepare the input image for YOLOv8 inference.

        This method performs several preprocessing steps on the input image:
        1. Extracts the image dimensions.
        2. Converts the image from BGR to RGB format.
        3. Resizes the image to match the expected input dimensions of the YOLOv8 model.
        4. Normalizes pixel values to be in the range [0, 1].
        5. Transposes the image dimensions to match the model's expectations.
        6. Expands the dimensions to match the model's batch size requirement.

        Args:
            image (numpy.ndarray): The input image to be processed.

        Returns:
            numpy.ndarray: The preprocessed image tensor ready for model inference.
        """

        # Extract height and width of the input image
        self.img_height, self.img_width = image.shape[:2]

        # Convert the image from BGR (common OpenCV format) to RGB
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to the expected input dimensions for the YOLOv8 model
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Normalize the pixel values to [0, 1]
        input_img = input_img / 255.0

        # Transpose the image dimensions (from HxWxC to CxHxW)
        input_img = input_img.transpose(2, 0, 1)

        # Expand dimensions to match the batch size requirement of the model
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        """
        Run inference on the given input tensor using the YOLOv8 model.

        This method feeds the preprocessed input tensor into the YOLOv8 model
        to get the object detection results.

        Args:
            input_tensor (numpy.ndarray): The preprocessed image tensor.

        Returns:
            list: Raw outputs from the YOLOv8 model.
        """

        # Feed the input tensor into the model and get the raw outputs
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs

    def process_output(self, output):
        """
        Process the raw output from the YOLOv8 model.

        This method takes the raw output from the model, extracts bounding boxes,
        confidence scores, and class IDs, and then applies non-maximum suppression
        to filter out overlapping and weak detections.

        Args:
            output (list): Raw outputs from the YOLOv8 model.

        Returns:
            list: Filtered bounding boxes.
            list: Confidence scores corresponding to the filtered bounding boxes.
            list: Class IDs corresponding to the filtered bounding boxes.
        """

        # Squeeze the output tensor and transpose it
        predictions = np.squeeze(output[0]).T

        # Extract and filter object confidence scores below the threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        # If no detections are above the confidence threshold, return empty lists
        if len(scores) == 0:
            return [], [], []

        # Identify the class with the highest confidence for each prediction
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Extract bounding boxes from the predictions
        boxes = self.extract_boxes(predictions)

        # Apply non-maximum suppression to remove overlapping and weak detections
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        """
        Extract and process bounding boxes from the model's predictions.

        This method extracts bounding boxes from the predictions, scales them
        to the original image dimensions, and converts them to the xyxy format.

        Args:
            predictions (np.array): Array containing predictions.
                                    The first four columns are assumed to be bounding box coordinates.

        Returns:
            np.array: Processed bounding boxes in xyxy format.
        """

        # Extract the first four columns from predictions which correspond to bounding box coordinates
        boxes = predictions[:, :4]

        # Scale the bounding box dimensions back to the size of the original image
        boxes = self.rescale_boxes(boxes)

        # Convert bounding box format from center (x, y) with width and height (w, h)
        # to top-left and bottom-right coordinates (x1, y1, x2, y2)
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        """
        Rescale bounding boxes to the original image dimensions.

        This method takes bounding boxes that are normalized to the model's input dimensions
        and scales them to the dimensions of the original image.

        Args:
            boxes (np.array): Array of bounding boxes normalized to the model's input dimensions.
                              Each box is represented as [x, y, width, height].

        Returns:
            np.array: Rescaled bounding boxes in the original image dimensions.
        """

        # Create an array representing the model's input dimensions repeated twice.
        # This is done because each bounding box has four values (x, y, width, height)
        # and we need to normalize and then rescale each of these values.
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])

        # Normalize the boxes using the model's input dimensions (normalizing each bounding box coordinate by dividing
        # it by the width or height of the model's input image dimensions.
        # The result will be values between 0 and 1, representing the relative position and size of each bounding box
        # with respect to the model's input dimensions.)
        boxes = np.divide(boxes, input_shape, dtype=np.float32)

        # Rescale the normalized boxes to the original image dimensions
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        return boxes

    def set_settings(self, coloring, confidence, displayAll, displayConfidence, displayLabel):
        """
        Update the settings for displaying detected objects.

        This method configures various display settings for detected objects, such as
        the confidence threshold, coloring scheme, and label display preferences.

        Args:
            coloring (str): Specifies the coloring scheme. Can be "multi-color" or "single-color".
            confidence (str): Confidence value or threshold. If above 1, it's converted to a percentage.
                              Values below 0.2 are set to a minimum threshold of 0.2.
            displayAll (bool): Flag to determine whether to display all detected objects.
            displayConfidence (bool): Flag to determine whether to display confidence scores with labels.
            displayLabel (bool): Flag to determine whether to display class labels with detections.

        Returns:
            None
        """

        # If a confidence value is provided
        if confidence != "":
            # Convert confidence to a fraction if it's represented as a percentage (greater than 1)
            if float(confidence) > 1:
                self.conf_threshold = float(confidence) / 100
            # Set a minimum confidence threshold if the provided value is too low
            elif float(confidence) < 0.2:
                self.conf_threshold = 0.2
            # Use the provided confidence value directly if it's in an appropriate range
            else:
                self.conf_threshold = float(confidence)

        # Set coloring scheme based on the provided coloring option
        if coloring == "multi-color":
            self.multi_color = True
        if coloring == "single-color":
            self.multi_color = False

        # Update settings for displaying detections
        self.displayAll = displayAll
        self.displayConfidence = displayConfidence
        self.displayLabel = displayLabel

    def draw_detections(self, image, required_class_ids=None, mask_alpha=0.2):
        """
        Draw detections on the given image.

        Args:
            image (numpy.ndarray): The input image on which detections will be drawn.
            required_class_ids (list or None): List of class IDs to restrict drawing only to specific classes. If None,
            all classes will be drawn.
            mask_alpha (float): Alpha value for the mask overlay transparency.

        Returns:
            numpy.ndarray: The image with detections drawn.
        """

        # Call the draw_detections function with provided parameters
        return draw_detections(
            image,
            self.boxes,
            self.scores,
            self.class_ids,
            required_class_ids,
            mask_alpha,
            self.multi_color,
            self.displayAll,
            self.displayConfidence,
            self.displayLabel
        )

    def get_input_details(self):
        """
        Get details about the input requirements of the model.

        This method retrieves information about the input tensors of the model,
        including their names and shapes, and stores this information as instance variables.
        """

        # Get the list of input details from the model session
        model_inputs = self.session.get_inputs()

        # Store the names of input tensors in self.input_names
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        # Store the shape of the first input tensor in self.input_shape
        self.input_shape = model_inputs[0].shape

        # Extract height and width dimensions from the input_shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        """
        Get details about the outputs requirements of the model.
        This method retrieves information about the output tensors of the model
        """

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]



