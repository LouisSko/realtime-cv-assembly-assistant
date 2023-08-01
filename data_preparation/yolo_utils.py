from imgaug import BoundingBox
import numpy as np
import os
import re
from PIL import Image


def read_yolo_file(file_path):
    """
    Read yolo file and output it as a list
    :param file_path:
    :return:
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        data = line.strip().split(' ')
        class_id = int(data[0])
        x_center = float(data[1])
        y_center = float(data[2])
        width = float(data[3])
        height = float(data[4])

        boxes.append([class_id, x_center, y_center, width, height])

    return boxes

def convert_yolo_to_bbf(boxes, pixel_height, pixel_width, formatBoundingBox=False):
    """
    Convert YOLO format to bounding box format
    (class_id, x_center, y_center, width, height) to (class id, x_min, x_max, y_min, y_max)
    """

    boxes_bbf = []

    for box in boxes:
        class_id, x_center, y_center, width, height = box

        # relative position
        x_min_rel = x_center - (width / 2)
        y_min_rel = y_center - (height / 2)
        x_max_rel = x_center + (width / 2)
        y_max_rel = y_center + (height / 2)

        # absolute position
        x_min = x_min_rel * pixel_width
        y_min = y_min_rel * pixel_height
        x_max = x_max_rel * pixel_width
        y_max = y_max_rel * pixel_height

        if formatBoundingBox:
            boxes_bbf.append(BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=class_id))
        else:
            boxes_bbf.append([x_min, y_min, x_max, y_max])

    return boxes_bbf


def convert_bbf_to_yolo(boxes):
    """
    Convert bounding box format to YOLO format
    (class id, x_min, x_max, y_min, y_max) to (class_id, x_center, y_center, width, height)
    """
    pixel_height, pixel_width = boxes.shape[0], boxes.shape[1]
    yolo_boxes = []
    for bb in boxes.bounding_boxes:

        # only store bb if its in the image
        if bb.is_fully_within_image(boxes.shape):
            class_id = bb.label
            x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2

            # only append boxes for those objects which are still visible after the augmentation
            # if not (x1 < 0 and x2 < 0) or (x1 > pixel_width and x2 > pixel_width) or (y1 < 0 and y2 < 0)or (y1 > pixel_height and y2 > pixel_height):

            x_center = (x1 + x2) / 2.0 / pixel_width
            y_center = (y1 + y2) / 2.0 / pixel_height
            width = (x2 - x1) / pixel_width
            height = (y2 - y1) / pixel_height

            yolo_boxes.append(
                [class_id, np.round(x_center, 6), np.round(y_center, 6), np.round(width, 6), np.round(height, 6)])

    return yolo_boxes


def save_yolo_file(file_path, bounding_boxes):
    with open(file_path, 'w') as file:
        for bbox in bounding_boxes:
            line = ' '.join(str(value) for value in bbox)
            file.write(line + '\n')


# creates a new directory inside directory which is called XML

def convert_yolo_to_voc(directory, labels_dir, dir_images, yolo_class_list_file):
    # Get a list of all the classes used in the yolo format
    with open(yolo_class_list_file) as f:
        yolo_classes = f.readlines()
    array_of_yolo_classes = [x.strip() for x in yolo_classes]


    # Description of Yolo Format values
    # 15 0.448743 0.529142 0.051587 0.021081
    # class_number x_yolo y_yolo yolo_width yolo_height

    def is_number(n):
        try:
            float(n)
            return True
        except ValueError:
            return False


    if not os.path.exists(os.path.join(directory, 'XML')):
        # If an XML folder does not already exist, make one
        os.mkdir(os.path.join(directory, 'XML'))

    for yolo_file in os.listdir(labels_dir):
        if yolo_file.endswith("txt"):
            the_file = open(os.path.join(labels_dir, yolo_file), 'r')
            all_lines = the_file.readlines()
            image_name = yolo_file

            file_name = yolo_file.split('.')[0]

            # Check to see if there is an image that matches the txt file
            if os.path.exists(os.path.join(dir_images, file_name + '.jpeg')):
                image_name = os.path.join(dir_images, file_name + '.jpeg')
            if os.path.exists(os.path.join(dir_images, file_name + '.jpg')):
                image_name = os.path.join(dir_images, file_name + '.jpg')
            if os.path.exists(os.path.join(dir_images, file_name + '.png')):
                image_name = os.path.join(dir_images, file_name + '.png')

            if not image_name == yolo_file:
                # If the image name is the same as the yolo filename
                # then we did NOT find an image that matches, and we will skip this code block
                orig_img = Image.open(image_name)  # open the image
                image_width = orig_img.width
                image_height = orig_img.height

                # Start the XML file
                with open(os.path.join(directory, 'XML', yolo_file.replace('txt', 'xml')), 'w') as f:
                    f.write('<annotation>\n')
                    f.write('\t<folder>XML</folder>\n')
                    f.write('\t<filename>' + image_name + '</filename>\n')
                    f.write('\t<path>' + os.getcwd() + os.sep + image_name + '</path>\n')
                    f.write('\t<source>\n')
                    f.write('\t\t<database>Unknown</database>\n')
                    f.write('\t</source>\n')
                    f.write('\t<size>\n')
                    f.write('\t\t<width>' + str(image_width) + '</width>\n')
                    f.write('\t\t<height>' + str(image_height) + '</height>\n')
                    f.write('\t\t<depth>3</depth>\n')  # assuming a 3 channel color image (RGB)
                    f.write('\t</size>\n')
                    f.write('\t<segmented>0</segmented>\n')

                    for each_line in all_lines:
                        # regex to find the numbers in each line of the text file
                        yolo_array = re.split("\s", each_line.rstrip())  # remove any extra space from the end of the line

                        # initalize the variables
                        class_number = 0
                        x_yolo = 0.0
                        y_yolo = 0.0
                        yolo_width = 0.0
                        yolo_height = 0.0
                        yolo_array_contains_only_digits = True

                        # make sure the array has the correct number of items
                        if len(yolo_array) == 5:
                            for each_value in yolo_array:
                                # If a value is not a number, then the format is not correct, return false
                                if not is_number(each_value):
                                    yolo_array_contains_only_digits = False

                            if yolo_array_contains_only_digits:
                                # assign the variables
                                class_number = int(yolo_array[0])
                                object_name = array_of_yolo_classes[class_number]
                                x_yolo = float(yolo_array[1])
                                y_yolo = float(yolo_array[2])
                                yolo_width = float(yolo_array[3])
                                yolo_height = float(yolo_array[4])

                                # Convert Yolo Format to Pascal VOC format
                                box_width = yolo_width * image_width
                                box_height = yolo_height * image_height
                                x_min = str(int(x_yolo * image_width - (box_width / 2)))
                                y_min = str(int(y_yolo * image_height - (box_height / 2)))
                                x_max = str(int(x_yolo * image_width + (box_width / 2)))
                                y_max = str(int(y_yolo * image_height + (box_height / 2)))

                                # write each object to the file
                                f.write('\t<object>\n')
                                f.write('\t\t<name>' + object_name + '</name>\n')
                                f.write('\t\t<pose>Unspecified</pose>\n')
                                f.write('\t\t<truncated>0</truncated>\n')
                                f.write('\t\t<difficult>0</difficult>\n')
                                f.write('\t\t<bndbox>\n')
                                f.write('\t\t\t<xmin>' + x_min + '</xmin>\n')
                                f.write('\t\t\t<ymin>' + y_min + '</ymin>\n')
                                f.write('\t\t\t<xmax>' + x_max + '</xmax>\n')
                                f.write('\t\t\t<ymax>' + y_max + '</ymax>\n')
                                f.write('\t\t</bndbox>\n')
                                f.write('\t</object>\n')

                    # Close the annotation tag once all the objects have been written to the file
                    f.write('</annotation>\n')
                    f.close()  # Close the file

    print("Conversion complete")

