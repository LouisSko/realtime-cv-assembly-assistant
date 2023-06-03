from PIL import Image
import piexif
import os
import pandas as pd


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


if __name__ == '__main__':
    dir_images = "/Users/louis.skowronek/aiss_images/images"
    remove_exif(dir_images)
