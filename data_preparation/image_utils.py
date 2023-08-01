import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import piexif


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



def remove_exif(dir_images):

    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    image_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(file.lower().endswith(ext) for ext in image_extensions)]

    df_img = pd.DataFrame(columns=['image_path', 'rotation'])

    for i, img in enumerate(image_files):
        filepath = os.path.join(dir_images, img)
        image = Image.open(filepath)

        # Get the EXIF data
        exif_data = image.info.get("exif")

        # Check if EXIF data exists
        if exif_data:
            # Convert the EXIF data to a mutable dictionary
            exif_dict = piexif.load(exif_data)
            # print(exif_dict.keys())
            # Check if the orientation tag exists
            if 274 in exif_dict["0th"]:
                # Update the orientation tag to 0 (normal)
                print(filepath)
                print(f'{i}: orientation is {exif_dict["0th"][274]} --- Set to 0')

                # Store information
                df_img.loc[i, ['image_path', 'rotation']] = [filepath, exif_dict["0th"][274]]

                # Set exif direction to 0
                exif_dict["0th"][274] = 0

                # Convert the EXIF data back to bytes
                exif_bytes = piexif.dump(exif_dict)

                # Save the image with updated EXIF data
                image.save(filepath) #, exif=exif_bytes)


            else:
                print("No orientation tag found.")

        else:
            print("No EXIF data found.")

    return df_img


