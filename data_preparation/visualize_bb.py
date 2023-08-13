import cv2
import os
from yolo_utils import read_yolo_file, convert_yolo_to_bbf
from PIL import Image, ImageDraw


def display_bb(dir_images, dir_annot, image_number=0, filename=None, replace=False, rotation_direction=0, width=4,
               show_image=False):
    """
    Display bounding boxes on an image and rotate them if necessary. Used for correcting the wrong BB due to EXIF orientation

    Args:
        dir_images (str): Directory containing the images.
        dir_annot (str): Directory containing the bounding box annotations.
        image_number (int, optional): Index of the image to display. Defaults to 0.
        filename (str, optional): Specific filename of the image to display. Overrides `image_number` if provided.
        replace (bool, optional): If True, replaces the original image with the displayed image. Defaults to False.
        rotation_direction (int, optional): Direction for image rotation. 1 for clockwise, 2 for counter-clockwise. Defaults to 0 (no rotation).
        width (int, optional): Width of the bounding box line. Defaults to 4.
        show_image (bool, optional): If True, displays the image with bounding boxes. Defaults to False.

    Returns:
        None
    """

    # List of valid image file extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    # Retrieve all valid image files from the directory
    image_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Determine the image to process
    if filename is not None:
        img = filename
    else:
        img = image_files[image_number]

    print(img)

    # Load the chosen image
    image_path = os.path.join(dir_images, img)
    image = cv2.imread(image_path)

    # Rotate the image if specified
    if rotation_direction == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_direction == 2:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Determine the annotation file corresponding to the image
    txt = img.split('.')[0] + '.txt'

    # Check if the annotation file exists
    if os.path.isfile(os.path.join(dir_annot, txt)):
        boxes = read_yolo_file(os.path.join(dir_annot, txt))

        # Convert the annotations from YOLO format to bounding box format
        boxes = convert_yolo_to_bbf(boxes, pixel_height=image.shape[0], pixel_width=image.shape[1])

        # Display the image with bounding boxes if specified
        if show_image:
            pil_image = Image.fromarray(image)

            # Initialize drawing context
            draw = ImageDraw.Draw(pil_image, mode="RGB")

            # Draw each bounding box on the image
            for box in boxes:
                draw.rectangle(box, outline="red", width=width)

            # Display the image's path on the image
            text_position = (10, 10)
            text_color = (255, 255, 255)
            draw.text(text_position, image_path, fill=text_color)

            pil_image.show()

        # Replace the original image with the processed one if specified
        if replace:
            cv2.imwrite(image_path, image)
            print('image saved')

    else:
        print(f'Annotation file {txt} not found!')


