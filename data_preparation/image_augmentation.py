import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters, BoundingBox, BoundingBoxesOnImage
from PIL import Image
import os
from tqdm import tqdm


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


def save_yolo_file(file_path, bounding_boxes):
    with open(file_path, 'w') as file:
        for bbox in bounding_boxes:
            line = ' '.join(str(value) for value in bbox)
            file.write(line + '\n')


def convert_yolo_to_bbf(boxes, pixel_height, pixel_width):
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

        boxes_bbf.append(BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=class_id))

    return boxes_bbf


def convert_bbf_to_yolo(boxes):
    """
    Convert bounding box format to YOLO format
    (class id, x_min, x_max, y_min, y_max) to (class_id, x_center, y_center, width, height)
    """
    pixel_height, pixel_width = boxes.shape[0], boxes.shape[1]
    yolo_boxes = []
    for bb in boxes.bounding_boxes:

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


def image_augmentation(dir_files, dir_augmentation, nr_of_augs=5):

    dir_images = os.path.join(dir_files, 'images')
    dir_annot = os.path.join(dir_files, 'labels')

    dir_aug_images = os.path.join(dir_augmentation, 'images')
    dir_aug_annot = os.path.join(dir_augmentation, 'labels')

    # Check if the directories exist, create them if necessary
    for dir_path in [dir_augmentation, dir_aug_images, dir_aug_annot]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Define augmentation sequence
    seq = augmenters.Sequential([
        augmenters.Fliplr(p=0.5),  # Horizontal flipping
        augmenters.Flipud(p=0.5),  # Vertical flipping
        # augmenters.AllChannelsCLAHE(clip_limit=(0.01, )),  # Contrast enhancement
        augmenters.Affine(rotate=(-45, 45)),  # Rotation between -30 to 30 degrees
        augmenters.AdditiveGaussianNoise(scale=(0, 0.6 * 255)),  # Add Gaussian noise
        augmenters.GammaContrast(gamma=(0.5, 2.0)),  # Gamma correction for contrast adjustment
        # augmenters.PerspectiveTransform(scale=(0.01, 0.1)),  # Perspective transformation
        augmenters.Multiply((0.7, 1.3)),  # Multiply pixel values by a random value between 0.5 and 1.5
        augmenters.Crop(px=(0, 1500)),  # Random cropping
    ])

    # List of image file extensions to consider
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]

    # Get all image files in the directory
    image_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Iterate over all files
    for img in tqdm(image_files):

        # get txt file
        txt = img.split('.')[0] + '.txt'

        # check if a .txt file exists
        if os.path.isfile(os.path.join(dir_annot, txt)):

            # Load original image and bounding box annotations
            image = cv2.imread(os.path.join(dir_images, img))
            original_image = Image.fromarray(image)
            boxes = read_yolo_file(os.path.join(dir_annot, txt))

            # Convert yolo format to bounding box format
            boxes_conv = convert_yolo_to_bbf(boxes, pixel_height=image.shape[0], pixel_width=image.shape[1])

            # Convert bounding box coordinates to imgaug format
            bbs = BoundingBoxesOnImage(boxes_conv, shape=image.shape)

            # Create augmentations for one image nr_of_augs times
            for it in range(nr_of_augs):

                # Apply augmentation
                aug_image, aug_bbs = seq(image=image, bounding_boxes=bbs)

                # Convert back to yolo - only store bbs which are fully visible in the image
                boxes_new = convert_bbf_to_yolo(aug_bbs)

                # Only store image, if it contains any boxes
                if len(boxes_new) > 0:
                    # Save image and annotations
                    aug_pil_image = Image.fromarray(aug_image)

                    # Generate unique file names for each augmentation
                    aug_image_name = f"{img.split('.')[0]}_{it}.jpg"
                    aug_annot_name = f"{img.split('.')[0]}_{it}.txt"

                    # save augmented image
                    aug_pil_image.save(os.path.join(dir_aug_images, aug_image_name))
                    save_yolo_file(os.path.join(dir_aug_annot, aug_annot_name), boxes_new)

            # save original image
            original_image.save(os.path.join(dir_aug_images, img))
            save_yolo_file(os.path.join(dir_aug_annot, txt), boxes)


if __name__ == "__main__":
    dir_files = '/Users/louis.skowronek/aiss_images'
    dir_augmentation = '/Users/louis.skowronek/aiss_images_augmented'
    nr_of_augs = 5

    image_augmentation(dir_files, dir_augmentation, nr_of_augs)

    # verbesserung: bboxes nicht direkt wegwerfen, wenn es diese nicht vollends im Bild ist
