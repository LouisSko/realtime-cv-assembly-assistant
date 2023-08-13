import os
import shutil
import random
from tqdm import tqdm
import argparse


def split_dataset(input_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split a dataset into training, testing, and validation sets based on given ratios.

    This function is used to organize images into train, test, and validation directories,
    according to the provided split ratios.

    Args:
        input_dir (str): Directory containing the dataset to be split.
        output_dir (str): Directory where the split dataset will be saved.
        split_ratio (tuple, optional): A tuple containing three float values representing
            the ratio of data to be used for training, testing, and validation respectively.
            The sum of the values should be 1.0. Defaults to (0.8, 0.1, 0.1).

    Returns:
        None
    """

    # Ensure the split ratio is valid
    assert sum(split_ratio) == 1.0, "Split ratio must sum up to 1.0"

    # Define directories for train, test, and validation sets
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Retrieve all images from the input directory
    images = os.listdir(os.path.join(input_dir, 'images'))

    # Randomly shuffle the images for a random split
    random.shuffle(images)

    # Determine the number of images for each subset based on the split ratios
    num_images = len(images)
    num_train = int(num_images * split_ratio[0])
    num_test = int(num_images * split_ratio[1])
    num_val = num_images - num_train - num_test

    # Split the list of images into respective subsets
    train_images = images[:num_train]
    test_images = images[num_train:num_train + num_test]
    val_images = images[num_train + num_test:]

    # Move the images (and corresponding labels if any) to their respective directories
    move_files(train_images, input_dir, train_dir)
    move_files(test_images, input_dir, test_dir)
    move_files(val_images, input_dir, val_dir)


def move_files(file_list, input_dir, output_dir):
    """
    Move image files and their corresponding label files to a new directory.

    This function is used to organize images and their corresponding labels into
    new directories, typically as a part of dataset splitting.

    Args:
        file_list (list): List of image filenames to be moved.
        input_dir (str): Directory containing the original dataset.
        output_dir (str): Directory where the files should be moved to.

    Returns:
        None
    """

    # Create directories for images and labels within the output directory
    output_dir_img = os.path.join(output_dir, 'images')
    output_dir_labels = os.path.join(output_dir, 'labels')
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    # Iterate over each file in the provided file list
    for file in tqdm(file_list):
        try:
            # Define source and destination paths for the image
            image_src = os.path.join(input_dir, 'images', file)
            image_dest = os.path.join(output_dir_img, file)

            # Define source and destination paths for the label
            label_file = file.split('.')[0] + '.txt'
            label_src = os.path.join(input_dir, 'labels', label_file)
            label_dest = os.path.join(output_dir_labels, label_file)

            # Copy the image and label to the output directory
            shutil.copy(image_src, image_dest)
            shutil.copy(label_src, label_dest)

        except Exception as e:
            # Print any errors encountered during file copying
            print(f'Error with file {image_src} or {label_file}: {e}')

def tuple_type(s):
    """
    Convert a comma-separated string to a tuple of floats.

    Args:
        s (str): The input string to be converted.

    Returns:
        tuple: Tuple containing the split values as floats.

    Raises:
        argparse.ArgumentTypeError: If the conversion is not successful.
    """
    try:
        return tuple(map(float, s.split(',')))
    except:
        raise argparse.ArgumentTypeError('Split ratios must be x,y,z')


# Check if the script is being run as the main module
if __name__ == "__main__":
    # Initialize an argument parser
    parser = argparse.ArgumentParser(description='Split dataset into train, test, and validation sets.')

    # Add argument for specifying the input directory
    parser.add_argument('--input_dir', type=str, default='/Users/louis.skowronek/AISS/generate_images',
                        help='Input directory path. Should contain folders "images" and "labels".')

    # Add argument for specifying the output directory
    parser.add_argument('--output_dir', type=str,
                        help='Output directory path. Default is the same as the input directory.')

    # Add argument for specifying the split ratio
    parser.add_argument('--split_ratio', type=tuple_type, default=(0.8, 0.1, 0.1),
                        help='Split ratio for train, test, and validation sets. Must be a comma-separated string.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # If the output directory is not specified, use the input directory as default
    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Call the function to split the dataset
    split_dataset(args.input_dir, args.output_dir, args.split_ratio)
