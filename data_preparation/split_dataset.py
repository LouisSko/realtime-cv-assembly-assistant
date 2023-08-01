import os
import shutil
import random
from tqdm import tqdm
import argparse


def split_dataset(input_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    assert sum(split_ratio) == 1.0, "Split ratio must sum up to 1.0"

    # Create the output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get the list of images in the input directory
    images = os.listdir(os.path.join(input_dir, 'images'))

    # Shuffle the images
    random.shuffle(images)

    # Calculate the number of images for each split
    num_images = len(images)
    num_train = int(num_images * split_ratio[0])
    num_test = int(num_images * split_ratio[1])
    num_val = num_images - num_train - num_test

    # Split the images into train, test, and val sets
    train_images = images[:num_train]
    test_images = images[num_train:num_train + num_test]
    val_images = images[num_train + num_test:]

    # Move the image and label files to the respective split directories
    move_files(train_images, input_dir, train_dir)
    move_files(test_images, input_dir, test_dir)
    move_files(val_images, input_dir, val_dir)


def move_files(file_list, input_dir, output_dir):
    # create directories
    output_dir_img = os.path.join(output_dir, 'images')
    output_dir_labels = os.path.join(output_dir, 'labels')
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)


    for file in tqdm(file_list):

        try:
            # Move the image file
            image_src = os.path.join(input_dir, 'images', file)
            image_dest = os.path.join(output_dir_img, file)

            label_file = file.split('.')[0] + '.txt'
            label_src = os.path.join(input_dir, 'labels', label_file)
            label_dest = os.path.join(output_dir_labels, label_file)

            shutil.copy(image_src, image_dest)
            shutil.copy(label_src, label_dest)

        except Exception as e:
            print(f'error at file {image_src}, {label_file}: {e}')

def tuple_type(s):
    try:
        return tuple(map(float, s.split(',')))
    except:
        raise argparse.ArgumentTypeError('Split ratios must be x,y,z')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train, test and validation sets.')
    parser.add_argument('--input_dir', type=str, default='/Users/louis.skowronek/AISS/generate_images',
                        help='Input directory path. Should contain folders "images" and "labels".')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory path. Default is the same as the input directory.')
    parser.add_argument('--split_ratio', type=tuple_type, default=(0.8, 0.1, 0.1),
                        help='Split ratio for train, test and validation sets. Must be a comma-separated string.')
    args = parser.parse_args()

    # If output_dir is not specified, use input_dir as default
    if args.output_dir is None:
        args.output_dir = args.input_dir

    split_dataset(args.input_dir, args.output_dir, args.split_ratio)







