from imgaug import augmenters, BoundingBox, BoundingBoxesOnImage
import numpy as np

def read_yolo_file(file_path):
    """
    Read yolo file and output it as a list
    :param file_path:
    :return:
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        data = line.strip().split(' ')
        class_id = int(data[0])
        x_center = float(data[1])
        y_center = float(data[2])
        width = float(data[3])
        height = float(data[4])

        boxes.append([class_id, x_center, y_center, width, height])

    return boxes

def convert_yolo_to_bbf(boxes, pixel_height, pixel_width, formatBoundingBox=False):
    """
    Convert YOLO format to bounding box format
    (class_id, x_center, y_center, width, height) to (class id, x_min, x_max, y_min, y_max)
    """

    boxes_bbf = []

    for box in boxes:
        class_id, x_center, y_center, width, height = box

        # relative position
        x_min_rel = x_center - (width / 2)
        y_min_rel = y_center - (height / 2)
        x_max_rel = x_center + (width / 2)
        y_max_rel = y_center + (height / 2)

        # absolute position
        x_min = x_min_rel * pixel_width
        y_min = y_min_rel * pixel_height
        x_max = x_max_rel * pixel_width
        y_max = y_max_rel * pixel_height

        if formatBoundingBox:
            boxes_bbf.append(BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=class_id))
        else:
            boxes_bbf.append([x_min, y_min, x_max, y_max])

    return boxes_bbf


def convert_bbf_to_yolo(BoundingBox, boxes):
    """
    Convert bounding box format to YOLO format
    (class id, x_min, x_max, y_min, y_max) to (class_id, x_center, y_center, width, height)
    """
    pixel_height, pixel_width = boxes.shape[0], boxes.shape[1]
    yolo_boxes = []
    for bb in BoundingBox.bounding_boxes:

        # only store bb if its in the image
        if bb.is_fully_within_image(boxes.shape):
            class_id = bb.label
            x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2

            # only append boxes for those objects which are still visible after the augmentation
            # if not (x1 < 0 and x2 < 0) or (x1 > pixel_width and x2 > pixel_width) or (y1 < 0 and y2 < 0)or (y1 > pixel_height and y2 > pixel_height):

            x_center = (x1 + x2) / 2.0 / pixel_width
            y_center = (y1 + y2) / 2.0 / pixel_height
            width = (x2 - x1) / pixel_width
            height = (y2 - y1) / pixel_height

            yolo_boxes.append(
                [class_id, np.round(x_center, 6), np.round(y_center, 6), np.round(width, 6), np.round(height, 6)])

    return yolo_boxes


def save_yolo_file(file_path, bounding_boxes):
    with open(file_path, 'w') as file:
        for bbox in bounding_boxes:
            line = ' '.join(str(value) for value in bbox)
            file.write(line + '\n')

