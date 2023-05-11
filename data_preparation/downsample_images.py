import os
from PIL import Image
from tqdm import tqdm

def resize_images(directory, output_directory, max_width, max_height):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all files in the directory
    for file_name in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):

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
            output_file_path = os.path.join(output_directory, file_name)
            resized_img.save(output_file_path)

if __name__=='__main__':

    # Input directory path
    directory_path = '/Users/louis.skowronek/aiss_images_augmented/images'
    # Output directory path
    output_directory_path = '/Users/louis.skowronek/aiss_images_augmented/images'

    # Target resolution (1080p)
    max_width = 1080
    max_height = 720

    # Resize the images
    resize_images(directory_path, output_directory_path, max_width, max_height)
