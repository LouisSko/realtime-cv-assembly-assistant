import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import piexif


def convert_image_to_png(image_dir, png_dir, drop_original=True):
    """
    Convert images in a directory to PNG format.

    This function takes images from the specified directory, converts them to PNG format,
    and saves the converted images in another directory. Optionally, it can also remove
    the original images after conversion.

    Args:
        image_dir (str): Directory containing the images to be converted.
        png_dir (str): Directory where the converted PNG images will be saved.
        drop_original (bool, optional): Whether to remove the original images after conversion. Defaults to True.

    Returns:
        None
    """

    # List of valid image file extensions to consider for conversion
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".heic"]

    # Retrieve all valid image files from the provided directory
    jpeg_images = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Iterate through each image and convert it to PNG format
    for jpeg_image in tqdm(jpeg_images):

        # Load the image using PIL
        image_path = os.path.join(image_dir, jpeg_image)
        image = Image.open(image_path)

        # Determine the name of the new PNG image
        png_image = os.path.splitext(jpeg_image)[0] + ".png"
        png_path = os.path.join(png_dir, png_image)

        # If the 'drop_original' flag is set, remove the original image
        if drop_original:
            os.remove(image_path)

        # Save the loaded image in PNG format
        image.save(png_path, "PNG")


def convert_image_to_jpeg(image_dir, jpeg_dir, drop_original=True):
    """
    Convert images in a directory to JPEG format.

    This function takes images from the specified directory, converts them to JPEG format,
    and saves the converted images in another directory. Optionally, it can also remove
    the original images after conversion.

    Args:
        image_dir (str): Directory containing the images to be converted.
        jpeg_dir (str): Directory where the converted JPEG images will be saved.
        drop_original (bool, optional): Whether to remove the original images after conversion. Defaults to True.

    Returns:
        None
    """

    # List of valid image file extensions to consider for conversion
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".heic"]

    # Retrieve all valid image files from the provided directory
    png_images = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Iterate through each image and convert it to JPEG format
    for png_image in tqdm(png_images):

        # Load the image using PIL
        image_path = os.path.join(image_dir, png_image)
        image = Image.open(image_path)

        # Determine the name of the new JPEG image
        jpeg_image = os.path.splitext(png_image)[0] + ".jpeg"
        jpeg_path = os.path.join(jpeg_dir, jpeg_image)

        # If the 'drop_original' flag is set, remove the original image
        if drop_original:
            os.remove(image_path)

        # Convert the image to RGB mode (necessary for JPEG) and save in JPEG format
        image.convert("RGB").save(jpeg_path, "JPEG")


def remove_exif(dir_images):
    """
    Removes the EXIF orientation data from images in the provided directory.

    This function iterates over images in the specified directory, checks for EXIF
    orientation data, and if found, removes it. The function also maintains a record
    of images that had their orientation data removed.

    Args:
        dir_images (str): Directory containing the images to process.

    Returns:
        pd.DataFrame: A DataFrame containing paths to images that had their orientation
        data removed and the original orientation value.
    """

    # Define the list of valid image file extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]

    # Retrieve all valid image files from the provided directory
    image_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Initialize a DataFrame to store paths and orientation values
    df_img = pd.DataFrame(columns=['image_path', 'rotation'])

    # Iterate over each image to process EXIF data
    for i, img in enumerate(image_files):
        filepath = os.path.join(dir_images, img)
        image = Image.open(filepath)

        # Retrieve the EXIF data of the image
        exif_data = image.info.get("exif")

        # If EXIF data exists, process it
        if exif_data:
            exif_dict = piexif.load(exif_data)

            # Check for the presence of the orientation tag
            if 274 in exif_dict["0th"]:
                print(filepath)
                print(f'{i}: orientation is {exif_dict["0th"][274]} --- Set to 0')

                # Record the image's path and its original orientation value
                df_img.loc[i, ['image_path', 'rotation']] = [filepath, exif_dict["0th"][274]]

                # Remove the orientation tag
                exif_dict["0th"][274] = 0

                # Convert the EXIF data back to byte format
                exif_bytes = piexif.dump(exif_dict)

                # Save the image without the orientation EXIF data
                image.save(filepath)  # , exif=exif_bytes)
            else:
                print("No orientation tag found.")
        else:
            print("No EXIF data found.")

    return df_img

