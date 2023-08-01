from ultralytics import YOLO
import cv2
import os
import numpy as np
from yolo_utils import save_yolo_file


def annotate_images(model, dir_images, dir_labels):
    # make label directory if necessary
    if not os.path.exists(dir_labels):
        print('Directory for labels does not exist')
        os.makedirs(dir_labels)
        print('Created directory for labels')

    # read in files
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".heic"]
    images_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # get annotations
    for file in images_files:
        image_path = os.path.join(dir_images, file)
        img = cv2.imread(image_path)
        results = model(img)

        # convert to yolo format
        yolo_boxes = []

        nr_boxes = len(results[0].boxes.cls)
        for i in range(0, nr_boxes):
            class_id = results[0].boxes.cls[i]
            x_min_rel, y_min_rel, x_max_rel, y_max_rel = results[0].boxes.xyxyn[i]
            width = x_max_rel - x_min_rel
            height = y_max_rel - y_min_rel
            x_center = x_min_rel + (width / 2)
            y_center = y_min_rel + (height / 2)
            yolo_boxes.append(
                [int(class_id.item()), np.round(x_center.item(), 6), np.round(y_center.item(), 6),
                 np.round(width.item(), 6), np.round(height.item(), 6)])

        annot_file = file.split('.')[0] + '.txt'
        annot_file_path = os.path.join(dir_labels, annot_file)
        save_yolo_file(annot_file_path, yolo_boxes)


if __name__ == "__main__":
    # load a pretrained model
    model = YOLO("/Users/louis.skowronek/Downloads/test/yolov8n_custom/weights/yolov8s_best.pt")
    # specify directories
    dir_images = '/Users/louis.skowronek/Downloads/generate_images/images'
    dir_annot_files = '/Users/louis.skowronek/Downloads/generate_images/labels'
    annotate_images(model, dir_images, dir_annot_files)
