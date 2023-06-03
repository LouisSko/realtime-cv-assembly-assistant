import os
from PIL import Image
from tqdm import tqdm


def convert_image_to_png(image_dir, png_dir, drop_original=True):
    # List of image file extensions to consider
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".heic"]
    # Get all image files in the directory
    jpeg_images = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Convert each JPEG image to PNG format
    for jpeg_image in tqdm(jpeg_images):
        # Load the JPEG image
        image_path = os.path.join(image_dir, jpeg_image)
        image = Image.open(image_path)

        # Convert the image to PNG format
        png_image = os.path.splitext(jpeg_image)[0] + ".png"
        png_path = os.path.join(png_dir, png_image)

        if drop_original:
            os.remove(image_path)

        image.save(png_path, "PNG")




def convert_image_to_jpeg(image_dir, jpeg_dir, drop_original=True):
    # List of image file extensions to consider
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".heic"]
    # Get all image files in the directory
    png_images = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Convert each PNG image to JPEG format
    for png_image in tqdm(png_images):
        # Load the PNG image
        image_path = os.path.join(image_dir, png_image)
        image = Image.open(image_path)

        # Convert the image to JPEG format
        jpeg_image = os.path.splitext(png_image)[0] + ".jpeg"
        jpeg_path = os.path.join(jpeg_dir, jpeg_image)

        if drop_original:
            os.remove(image_path)

        image.convert("RGB").save(jpeg_path, "JPEG")


