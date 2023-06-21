import os
import shutil
import random
from tqdm import tqdm


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


if __name__ == "__main__":
    # Input directory path. Should contain a folder images and labels
    input_directory = '/Users/louis.skowronek/AISS/aiss_images'

    # Output directory path
    output_directory = '/Users/louis.skowronek/AISS/aiss_images'

    # Split ratio (train, test, val)
    split_ratio = (0.8, 0.1, 0.1)

    # Split the dataset
    split_dataset(input_directory, output_directory, split_ratio)
