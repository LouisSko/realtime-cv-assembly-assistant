import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters, BoundingBox, BoundingBoxesOnImage
import os
from tqdm import tqdm
from convert_yolo import read_yolo_file, convert_yolo_to_bbf, convert_bbf_to_yolo, save_yolo_file

from itertools import chain
from imutils import paths
from multiprocessing import Pool

def augment(img, seq):
    
        # Define augmentation sequence
    seq = augmenters.Sequential([
        augmenters.Fliplr(p=0.5),  # Horizontal flipping
        augmenters.Flipud(p=0.5),  # Vertical flipping
        augmenters.Affine(rotate=(-45, 45)),
        augmenters.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Add Gaussian 
        augmenters.Crop(percent=(0, 0.1)),  # Random cropping
    ])
    
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

                               
def image_augmentation_parallel():

    
    # 
    dir_images = '/pfs/data5/home/kit/stud/ulhni/aiss/aiss_images/train'

    # List of image file extensions to consider
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]

    # Get all image files in the directory
    image_files = paths.list_images(dir_images)
    pool = Pool(10)
    pool.map(augment, image_files)
    
    
    
    