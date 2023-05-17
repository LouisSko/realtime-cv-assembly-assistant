import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters, BoundingBox, BoundingBoxesOnImage
import os
from tqdm import tqdm
from convert_yolo import read_yolo_file, convert_yolo_to_bbf, convert_bbf_to_yolo, save_yolo_file


def image_augmentation(input_dir, output_dir, nr_of_augs=5):

    dir_images = os.path.join(input_dir, 'images')
    dir_annot = os.path.join(input_dir, 'labels')

    dir_aug_images = os.path.join(output_dir, 'images')
    dir_aug_annot = os.path.join(output_dir, 'labels')

    # Check if the directories exist, create them if necessary
    for dir_path in [output_dir, dir_aug_images, dir_aug_annot]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Define augmentation sequence
    seq = augmenters.Sequential([
        augmenters.Fliplr(p=0.5),  # Horizontal flipping
        augmenters.Flipud(p=0.5),  # Vertical flipping
        # augmenters.AllChannelsCLAHE(clip_limit=(0.01, )),  # Contrast enhancement
        augmenters.Affine(rotate=(-45, 45)),  # Rotation between -30 to 30 degrees
        # augmenters.AdditiveGaussianNoise(scale=(0, 0.6 * 255)),  # Add Gaussian noise
        # augmenters.GammaContrast(gamma=(0.5, 2.0)),  # Gamma correction for contrast adjustment
        # augmenters.PerspectiveTransform(scale=(0.01, 0.1)),  # Perspective transformation
        # augmenters.Multiply((0.7, 1.3)),  # Multiply pixel values by a random value between 0.5 and 1.5
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

            boxes = read_yolo_file(os.path.join(dir_annot, txt))

            # Convert yolo format to bounding box format
            boxes_conv = convert_yolo_to_bbf(boxes, pixel_height=image.shape[0], pixel_width=image.shape[1], formatBoundingBox=True)

            # Convert bounding box coordinates to imgaug format
            bbs = BoundingBoxesOnImage(boxes_conv, shape=image.shape)

            # Create augmentations for one image nr_of_augs times
            for it in range(nr_of_augs):

                # Apply augmentation
                aug_image, aug_bbs = seq(image=image, bounding_boxes=bbs)
                # aug_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)

                # Convert back to yolo - only store bbs which are fully visible in the image
                boxes_new = convert_bbf_to_yolo(aug_bbs)

                # Only store image, if it contains any boxes
                if len(boxes_new) > 0:

                    # Generate unique file names for each augmentation
                    aug_image_name = f"{img.split('.')[0]}_{it}.jpeg"
                    aug_annot_name = f"{img.split('.')[0]}_{it}.txt"

                    # save augmented image
                    cv2.imwrite(os.path.join(dir_aug_images, aug_image_name), aug_image)
                    save_yolo_file(os.path.join(dir_aug_annot, aug_annot_name), boxes_new)

            # save original image
            # original_image.save(os.path.join(dir_aug_images, img))
            # save_yolo_file(os.path.join(dir_aug_annot, txt), boxes)


if __name__ == "__main__":

    # Input directory path. Should contain a folder images and labels
    input_dir = '/Users/louis.skowronek/aiss_images/train'

    # Output directory path
    output_dir = '/Users/louis.skowronek/aiss_images/train'

    # Number of augmented images per image
    nr_of_augs = 10

    image_augmentation(input_dir, output_dir, nr_of_augs)

    # verbesserung: bboxes nicht direkt wegwerfen, wenn es diese nicht vollends im Bild ist
