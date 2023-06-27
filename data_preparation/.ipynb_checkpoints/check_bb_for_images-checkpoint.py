import cv2
import os
from convert_yolo import read_yolo_file, convert_yolo_to_bbf
from PIL import Image, ImageDraw


def check_images(dir_images, dir_annot, image_number=0, filename=None, replace=False, rotation_direction=0, width=4, show_image=False):

    image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
    image_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(file.lower().endswith(ext) for ext in image_extensions)]

    if filename is not None:
        img = filename
    else:
        img = image_files[image_number]

    print(img)

  # Load original image and bounding box annotations
    image_path = os.path.join(dir_images, img)

    image = cv2.imread(image_path)

    # Rotate the image based on the provided direction
    if rotation_direction == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_direction == 2:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # get txt file
    txt = img.split('.')[0] + '.txt'

    # check if a .txt file exists
    if os.path.isfile(os.path.join(dir_annot, txt)):
        boxes = read_yolo_file(os.path.join(dir_annot, txt))

        # Convert yolo format to bounding box format
        boxes = convert_yolo_to_bbf(boxes, pixel_height=image.shape[0], pixel_width=image.shape[1])

        if show_image:
            pil_image = Image.fromarray(image)

            # Create an ImageDraw object
            draw = ImageDraw.Draw(pil_image, mode="RGB")

            # Draw each bounding box rectangle
            for box in boxes:
                draw.rectangle(box, outline="red", width=width)

            # add filename
            text_position = (10, 10)  # Specify the position where the text should be displayed
            text_color = (255, 255, 255)  # Specify the RGB color for the text
            draw.text(text_position, image_path, fill=text_color)

            pil_image.show()

        if replace:
            cv2.imwrite(image_path, image)
            print('image saved')

    else:
        print(f'file {txt} not found!')

