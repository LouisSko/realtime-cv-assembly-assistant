import cv2
from imgaug import augmenters, BoundingBoxesOnImage
import os
from tqdm import tqdm
from yolo_utils import read_yolo_file, convert_yolo_to_bbf, convert_bbf_to_yolo, save_yolo_file
import argparse


def image_augmentation(input_dir, output_dir, nr_of_augs=5):
    """
    Apply image augmentation to a dataset of images and their corresponding YOLO annotations.

    This function reads images and their YOLO annotations, applies a series of augmentations,
    and saves the augmented images and updated annotations to the specified output directory.

    :param input_dir (str): Directory containing the 'images' and 'labels' subdirectories with original data.
    :param output_dir (str): Directory where augmented 'images' and 'labels' will be saved.
    :param nr_of_augs (int): Number of augmentations to create for each image.
    :return: None
    """

    # Define paths for original images and annotations
    dir_images = os.path.join(input_dir, 'images')
    dir_annot = os.path.join(input_dir, 'labels')

    # Define paths for augmented images and annotations
    dir_aug_images = os.path.join(output_dir, 'images')
    dir_aug_annot = os.path.join(output_dir, 'labels')

    # Check if the output directories exist, create them if necessary
    for dir_path in [output_dir, dir_aug_images, dir_aug_annot]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Define augmentation sequence for images
    seq = augmenters.Sequential([
        augmenters.Fliplr(p=0.5),
        augmenters.Flipud(p=0.5),
        augmenters.Affine(rotate=(-45, 45)),
        augmenters.GammaContrast(gamma=(0.5, 2.0)),
        augmenters.Multiply((0.8, 1.2))
    ])

    # List of valid image file extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]

    # Get list of image files in the directory
    image_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Iterate over all image files
    for img in tqdm(image_files):

        # Determine corresponding annotation file for the image
        txt = img.split('.')[0] + '.txt'

        # Check if annotation file exists for the image
        if os.path.isfile(os.path.join(dir_annot, txt)):

            # Load the original image and its bounding box annotations
            image = cv2.imread(os.path.join(dir_images, img))
            boxes = read_yolo_file(os.path.join(dir_annot, txt))

            # Convert YOLO format bounding boxes to imgaug's format
            boxes_conv = convert_yolo_to_bbf(boxes, pixel_height=image.shape[0], pixel_width=image.shape[1],
                                             formatBoundingBox=True)
            bbs = BoundingBoxesOnImage(boxes_conv, shape=image.shape)

            # Apply augmentations for each image 'nr_of_augs' times
            for it in range(nr_of_augs):

                # Apply the defined augmentation sequence
                aug_image, aug_bbs = seq(image=image, bounding_boxes=bbs)

                # Convert the augmented bounding boxes back to YOLO format
                boxes_new = convert_bbf_to_yolo(aug_bbs)

                # Save augmented image and annotations only if there are valid bounding boxes
                if len(boxes_new) > 0:
                    aug_image_name = f"{img.split('.')[0]}_{it}.jpeg"
                    aug_annot_name = f"{img.split('.')[0]}_{it}.txt"
                    cv2.imwrite(os.path.join(dir_aug_images, aug_image_name), aug_image)
                    save_yolo_file(os.path.join(dir_aug_annot, aug_annot_name), boxes_new)

# Check if the script is being run as the main module
if __name__ == "__main__":

    # Initialize an argument parser
    parser = argparse.ArgumentParser(description="Augment Images")

    # Add argument for specifying the input directory which should contain 'images' and 'labels' subdirectories
    parser.add_argument("--input_dir", default="/Users/louis.skowronek/AISS/aiss_images/train",
                        help="Input directory path. Should contain a folder images and labels")

    # Add argument for specifying the output directory
    parser.add_argument("--output_dir", default=None,
                        help="Output directory path. If None, it will be the same as input_dir_base")

    # Add argument for specifying the number of augmentations per image
    parser.add_argument("--nr_of_augs", type=int, default=10,
                        help="Number of augmented images per image")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if an output directory is specified. If not, set it to be the same as the input directory
    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Call the image augmentation function with the parsed command-line arguments
    image_augmentation(args.input_dir, args.output_dir, args.nr_of_augs)
