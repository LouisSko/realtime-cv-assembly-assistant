from ultralytics import YOLO
import os
from yolo_utils import convert_yolo_to_voc, save_yolo_file
import sys
import cv2
import argparse
import numpy as np

def extract_frames(path_video, dir_save=None, frames_per_second=1):
    """
    Generate images from a given video file at a specified rate.

    Args:
        dir_video (str): Directory where the video file is located.
        file (str): Video file name.
        dir_save (str, optional): Directory to save the generated images.
            If None, images will be saved in the same directory as the video. Defaults to None.
        frames_per_second (int, optional): Number of frames to capture per second of video. Defaults to 1.
    """

    # Create a video capture object
    cap = cv2.VideoCapture(path_video)

    # extract video file name
    file = path_video.split('/')[-1]

    # Get frames per second of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare the images directory, create if it doesn't exist
    img_directory = dir_save
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)

    it = 0  # Frame iterator
    f = 0  # Image file count

    print(f'Generate images. Check {img_directory}')

    # Loop to capture images from video
    while cap.isOpened():
        success, img = cap.read()  # Read a frame from the video

        # If the frame read is not successful, break the loop
        if not success:
            break

        # Write the image to file every 'frames_per_second' frames
        if success and (it % (fps*frames_per_second) == 0):
            file_name = file.split('.')[0] + '_' + str(f) + '.jpeg'  # Create image file name
            # Write the image to file
            cv2.imwrite(os.path.join(img_directory, file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            f += 1  # Increment image file count

        it += 1  # Increment frame iterator

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the video capture object
    cap.release()

    print(f'images generated')



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



def main(args):
    # Check if save directory exists
    if os.path.exists(args.save_dir):
        print(f"Directory {args.save_dir} already exists. All content in this folder will be overwritten.")
        user_input = input("Do you want to continue? (yes/exit) ")

        if user_input.lower() == "yes":
            pass
        elif user_input.lower() == "exit":
            sys.exit("Process terminated by the user.")

    # Define subdirectories for storing images and labels
    images_dir = os.path.join(args.save_dir, 'images')
    labels_dir = os.path.join(args.save_dir, 'labels')

    # Generate images from video.
    extract_frames(args.path_video, dir_save=images_dir, frames_per_second=2)  # saves one image every second

    # Annotate images
    model = YOLO(args.model_path)
    annotate_images(model, images_dir, labels_dir)

    # Convert yolo files to voc
    # classes file need to be copied into generate_images folder
    convert_yolo_to_voc(args.save_dir, labels_dir, images_dir, args.yolo_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the paths.')
    parser.add_argument('--path_video',
                        default='/Users/louis.skowronek/object-detection-project/videos/black_axle_and_black_beam.mp4')
    parser.add_argument('--save_dir', default='/Users/louis.skowronek/AISS/test_files')
    parser.add_argument('--yolo_classes',
                        default='/Users/louis.skowronek/object-detection-project/onnx_yolov8/classes.txt')
    parser.add_argument('--model_path', default='../models/yolov8s_best.pt')
    args = parser.parse_args()

    main(args)
