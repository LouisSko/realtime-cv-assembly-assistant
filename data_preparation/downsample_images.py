import os
from PIL import Image
from tqdm import tqdm


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


if __name__ == '__main__':

    # Input directory path
    input_directory_path_base = '/Users/louis.skowronek/AISS/aiss_images'

    # Output directory path
    output_directory_path_base = '/Users/louis.skowronek/AISS/aiss_images'

    # Target resolution (1080p)
    max_width = 1080
    max_height = 720

    # resize images in all three folders
    for folder in ['train', 'val', 'test']:
        input_dir = os.path.join(input_directory_path_base, folder, 'images')
        output_dir = os.path.join(output_directory_path_base, folder, 'images')

        # Resize the images
        resize_images(input_dir, output_dir, max_width, max_height)
