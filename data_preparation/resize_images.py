import os
from PIL import Image
from tqdm import tqdm
import argparse

def resize_images(input_dir, output_dir, max_width, max_height):

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the directory
    for file_name in tqdm(os.listdir(input_dir)):

        file_path = os.path.join(input_dir, file_name)

        # Check file extension
        if os.path.isfile(file_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):

            # Open the image file
            img = Image.open(file_path)

            # Calculate the new dimensions while maintaining the aspect ratio
            width, height = img.size
            aspect_ratio = width / height

            new_width = min(width, max_width)
            new_height = min(height, max_height)

            if aspect_ratio > 1:
                new_height = int(new_width / aspect_ratio)
            else:
                new_width = int(new_height * aspect_ratio)

            # Resize the image
            resized_img = img.resize((new_width, new_height))

            # Save the resized image
            output_file_path = os.path.join(output_dir, file_name)
            resized_img.save(output_file_path)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Resize Images")

    parser.add_argument("--input_dir_base", default="/Users/louis.skowronek/AISS/generate_images/yolo_images",
                        help="Base input directory path")
    parser.add_argument("--output_dir_base", default=None,
                        help="Base output directory path. If None, it will be the same as input_dir_base")
    parser.add_argument("--max_width", type=int, default=1280,
                        help="Maximum width for the resized images")
    parser.add_argument("--max_height", type=int, default=720,
                        help="Maximum height for the resized images")

    args = parser.parse_args()

    # If no output directory is provided, use the input directory as the output directory
    if args.output_dir_base is None:
        args.output_dir_base = args.input_dir_base

    # Loop through each folder and resize the images
    for folder in ['train', 'val', 'test']:
        input_dir = os.path.join(args.input_dir_base, folder, 'images')
        output_dir = os.path.join(args.output_dir_base, folder, 'images')

        # Resize the images
        resize_images(input_dir, output_dir, args.max_width, args.max_height)
