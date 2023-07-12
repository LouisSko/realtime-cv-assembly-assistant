import numpy as np
import cv2
import os


#define the labels

# Get the directory path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the file you want to access
labels_path = os.path.join(current_dir, 'classes.txt')

#labels_path = '/object-detection-project/onnx_yolov8/yolov8/classes.txt'
try:
    class_names = [name.strip() for name in open(labels_path).readlines()]
except:
    print(f"Error while reading {labels_path}. Go into utils.py to change the path to classes.txt")

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, required_class_ids = None, mask_alpha=0.3):

    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.002)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        class_name = class_names[class_id]

        if class_name in required_class_ids or required_class_ids is None:
            color = colors[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (102, 102, 255), 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), (102, 102, 255), -1)

            if required_class_ids is None:
                label = class_name
                caption = f'{label} {int(score * 100)}%'
            else:
                caption = "Required piece"
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                        (x1 + tw, y1 - th), (0, 0, 255), -1)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), (0, 0, 255), -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


class MotionDetector:
    def __init__(self, threshold=30, th_diff=0.3, skip_frames=30):
        self.threshold = threshold
        self.th_diff = th_diff
        self.skip_frames = skip_frames
        self.prev_frame = None
        self.comparison_frame_counter = 0
        self.motion_flag = True

    def set_skip_frames(self):
        if self.motion_flag:
            self.skip_frames = 15
        else:
            self.skip_frames = 15

    def detect_motion(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return False

        self.comparison_frame_counter += 1

        if self.comparison_frame_counter >= self.skip_frames:
            # reset comparison counter
            self.comparison_frame_counter = 0

            # Convert frames to grayscale
            prev_frame_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform absolute difference between the current and previous frames
            frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)

            # Apply thresholding to obtain a binary image
            _, thresholded_diff = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

            # Count the number of non-zero pixels (white pixels) in the thresholded difference image
            motion_pixels = cv2.countNonZero(thresholded_diff)

            # calculate threshold
            total_pixels = self.prev_frame.shape[0] * self.prev_frame.shape[1]
            motion_pixels = cv2.countNonZero(thresholded_diff)

            # update prev_frame
            self.prev_frame = frame

            # if more than th_diff of the pixels are different return true
            if motion_pixels / total_pixels > self.th_diff:
                self.motion_flag = True
            else:
                self.motion_flag = False

        return self.motion_flag



def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3246,
    capture_height=1848,
    # capture_width=1280,
    # capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=21,
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

