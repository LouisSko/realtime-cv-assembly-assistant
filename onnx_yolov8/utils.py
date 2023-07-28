import numpy as np
import cv2
import os



# Get the directory path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the file you want to access
labels_path = os.path.join(current_dir, 'classes.txt')

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


def draw_detections(image, boxes, scores, class_ids, required_class_ids=None, mask_alpha=0.3, multi_color=False, displayAll=False, displayConfidence=False, displayLabel=False):

    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0005
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        label = class_names[class_id]

        # Check if every part is displayed or only required ones
        if (required_class_ids is not None and label in required_class_ids) or (required_class_ids is None) or (displayAll is True):
            x1, y1, x2, y2 = box.astype(int)

            # Settings
            if multi_color == False:
                if displayAll is True and required_class_ids is not None and label not in required_class_ids:
                     # Set color
                    color = (179, 179, 179)
                else:
                    # Set color
                    color = (102, 102, 255)
                
                # Draw rectangle
                cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
                # Draw fill rectangle in mask image
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                # Set color
                color = colors[class_id]
                # Draw rectangle
                cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
                # Draw fill rectangle in mask image
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            
            # Labels based on settings
            if displayConfidence == True and displayLabel == True:
                caption = f'{label} {int(score * 100)}%'
            elif  displayLabel == True and displayConfidence == False:
                caption = f'{label}'
            elif displayConfidence == True and displayLabel == False:
                caption = f'Required piece: {int(score * 100)}%'
            else:
                caption = "Required piece"

            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
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


def get_labels_steps():
    labels = {i: class_name for i, class_name in enumerate(class_names)}

    steps_no = {
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

    steps = {step: [labels[i] for i in num_list] for step, num_list in steps_no.items()}

    #output
    #steps = {
    #    1: [labels[8], labels[4], labels[15]],             # [red_oct_con, grey_axle_long, engine]
    #    2: [labels[3]],                                   # [grey_axle_short]
    #    3: [labels[0], labels[10]],                       # [grey_beam_bent, blue_pin_3L]
    #    4: [labels[10], labels[10], labels[13]],          # [blue_pin_3L, blue_pin_3L, white_beam_L]
    #    5: [labels[11], labels[11], labels[12]],          # [blue_axle_pin, blue_axle_pin, white_beam_bent]
    #    6: [labels[1]],                                   # [grey_axle_long_stop]
    #    7: [labels[6]],                                   # [black_beam]
    #    8: [labels[7], labels[7]],                        # [black_pin_short, black_pin_short]
    #    9: [labels[16]],                                  # [wheel]
    #    10: [labels[14]],                                 # [white_tooth]
    #    11: [labels[2]],                                  # [grey_axle_short_stop]
    #    12: [labels[9], labels[9]],                       # [red_pin_3L, red_pin_3L]
    #    13: [labels[11], labels[11], labels[5], labels[7]],# [blue_axle_pin, blue_axle_pin, black_axle_pin_con, black_pin_short]
    #    14: [labels[11], labels[11], labels[5], labels[7]],# [blue_axle_pin, blue_axle_pin, black_axle_pin_con, black_pin_short]
    #    15: [labels[17]]                                  # [wire]
    #}

    return labels, steps_no, steps

