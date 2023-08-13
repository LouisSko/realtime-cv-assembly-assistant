import os
from PIL import Image
from tqdm import tqdm
import argparse


def resize_images(input_dir, output_dir, max_width, max_height):
    """
    Resizes images from the input directory and saves them in the output directory.

    This function maintains the aspect ratio of each image and ensures that the new
    dimensions do not exceed the provided maximum width and height.

    Args:
        input_dir (str): Directory containing the images to be resized.
        output_dir (str): Directory where the resized images will be saved.
        max_width (int): Maximum allowed width for the resized images.
        max_height (int): Maximum allowed height for the resized images.

    Returns:
        None
    """

    # Ensure the output directory exists, create if necessary
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input directory
    for file_name in tqdm(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)

        # Only process valid image file formats
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):

            # Load the image using PIL
            img = Image.open(file_path)

            # Calculate the aspect ratio of the image
            width, height = img.size
            aspect_ratio = width / height

            # Determine new dimensions based on the aspect ratio and max dimensions
            new_width = min(width, max_width)
            new_height = min(height, max_height)

            if aspect_ratio > 1:
                new_height = int(new_width / aspect_ratio)
            else:
                new_width = int(new_height * aspect_ratio)

            # Resize the image to the new dimensions
            resized_img = img.resize((new_width, new_height))

            # Save the resized image to the output directory
            output_file_path = os.path.join(output_dir, file_name)
            resized_img.save(output_file_path)


# Check if the script is being run as the main module
if __name__ == "__main__":

    # Initialize an argument parser for command-line input
    parser = argparse.ArgumentParser(description="Resize Images")

    # Add argument for specifying the base input directory
    parser.add_argument("--input_dir_base", default="/Users/louis.skowronek/AISS/generate_images/yolo_images",
                        help="Base input directory path")

    # Add argument for specifying the base output directory
    parser.add_argument("--output_dir_base", default=None,
                        help="Base output directory path. If None, it will be the same as input_dir_base")

    # Add argument for specifying the maximum width of resized images
    parser.add_argument("--max_width", type=int, default=1280,
                        help="Maximum width for the resized images")

    # Add argument for specifying the maximum height of resized images
    parser.add_argument("--max_height", type=int, default=720,
                        help="Maximum height for the resized images")

    # Parse the command-line arguments
    args = parser.parse_args()

    # If no base output directory is provided, use the base input directory for saving resized images
    if args.output_dir_base is None:
        args.output_dir_base = args.input_dir_base

    # Process images in the train, val, and test folders
    for folder in ['train', 'val', 'test']:
        input_dir = os.path.join(args.input_dir_base, folder, 'images')
        output_dir = os.path.join(args.output_dir_base, folder, 'images')

        # Call the function to resize images
        resize_images(input_dir, output_dir, args.max_width, args.max_height)
