import os
from PIL import Image
from tqdm import tqdm


def convert_jpeg_to_png(jpeg_dir, png_dir, drop_original=True):
    # List of image file extensions to consider
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    # Get all image files in the directory
    jpeg_images = [file for file in os.listdir(jpeg_dir) if os.path.isfile(os.path.join(jpeg_dir, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Convert each JPEG image to PNG format
    for jpeg_image in tqdm(jpeg_images):
        # Load the JPEG image
        image_path = os.path.join(jpeg_dir, jpeg_image)
        image = Image.open(image_path)

        # Convert the image to PNG format
        png_image = os.path.splitext(jpeg_image)[0] + ".png"
        png_path = os.path.join(png_dir, png_image)

        image.save(png_path, "PNG")

        if drop_original:
            os.remove(image_path)

