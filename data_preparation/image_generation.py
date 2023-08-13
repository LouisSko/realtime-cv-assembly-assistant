from ultralytics import YOLO
import os
from yolo_utils import convert_yolo_to_voc, save_yolo_file
import sys
import cv2
import argparse
import numpy as np


def extract_frames(path_video, dir_save=None, frames_per_second=1):
    """
    Extracts and saves frames from a video at a specified rate.

    Args:
        path_video (str): Full path to the video file from which frames will be extracted.
        dir_save (str, optional): Directory where the extracted frames will be saved.
            If not provided, frames will be saved in the current directory. Defaults to None.
        frames_per_second (int, optional): Number of frames to capture per second of video. Defaults to 1.

    Returns:
        None
    """

    # Initialize a video capture object for the given video file
    cap = cv2.VideoCapture(path_video)

    # Extract the video file name from the provided path
    file = path_video.split('/')[-1]

    # Determine the frames per second of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # If a save directory isn't specified, set it to the current directory
    if dir_save is None:
        dir_save = os.getcwd()

    # Ensure the save directory exists; if not, create it
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    # Initialize counters for the frame iteration and saved image file count
    it, f = 0, 0

    print(f'Extracting frames. The frames will be saved in: {dir_save}')

    # Loop through the video to capture frames
    while cap.isOpened():
        success, img = cap.read()  # Read the next frame from the video

        # If reading the frame isn't successful, exit the loop
        if not success:
            break

        # If the current frame is one of the desired frames per second, save it
        if success and (it % (fps // frames_per_second) == 0):
            file_name = f"{file.split('.')[0]}_{f}.jpeg"  # Generate the file name for the frame
            cv2.imwrite(os.path.join(dir_save, file_name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])  # Save the frame as a JPEG image
            f += 1  # Increment the saved image counter

        it += 1  # Increment the frame counter

        # Provide an option to stop the frame extraction early by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object to free up resources
    cap.release()

    print(f'{f} frames extracted and saved.')


def annotate_images(model, dir_images, dir_labels):
    """
    Annotate images using the provided model and save annotations in YOLO format.

    Given a directory of images and a trained model, this function processes each
    image to detect objects and then saves the detected objects' bounding boxes in
    YOLO format.

    Args:
        model: A trained model capable of object detection.
        dir_images (str): Directory containing the images to be annotated.
        dir_labels (str): Directory where the generated YOLO annotations will be saved.

    Returns:
        None
    """

    # Check if the specified directory for labels exists, if not, create it
    if not os.path.exists(dir_labels):
        print('Directory for labels does not exist')
        os.makedirs(dir_labels)
        print('Created directory for labels')

    # Define the valid image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".heic"]

    # Get the list of image files in the provided directory
    images_files = [file for file in os.listdir(dir_images) if os.path.isfile(os.path.join(dir_images, file)) and any(
        file.lower().endswith(ext) for ext in image_extensions)]

    # Iterate over each image file to generate annotations
    for file in images_files:
        image_path = os.path.join(dir_images, file)
        img = cv2.imread(image_path)

        # Use the model to detect objects in the image
        results = model(img)

        # Convert detected bounding boxes to YOLO format
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

        # Save the annotations in YOLO format
        annot_file = file.split('.')[0] + '.txt'
        annot_file_path = os.path.join(dir_labels, annot_file)
        save_yolo_file(annot_file_path, yolo_boxes)


def main(args):
    """
    Process a video to extract frames, annotate the frames using a YOLO model,
    and then convert the YOLO annotations to VOC format.

    Args:
        args: Command line arguments containing the following attributes:
            - save_dir (str): Directory where the processed images and annotations will be saved.
            - path_video (str): Path to the video file to process.
            - model_path (str): Path to the trained YOLO model for object detection.
            - yolo_classes (str): Path to the YOLO classes file.
    """

    # Check if the specified save directory already exists
    if os.path.exists(args.save_dir):
        print(f"Directory {args.save_dir} already exists. All content in this folder will be overwritten.")

        # Prompt the user for confirmation before overwriting the existing directory
        user_input = input("Do you want to continue? (yes/exit) ")

        if user_input.lower() == "yes":
            pass
        elif user_input.lower() == "exit":
            sys.exit("Process terminated by the user.")

    # Define subdirectories to store extracted images and their annotations
    images_dir = os.path.join(args.save_dir, 'images')
    labels_dir = os.path.join(args.save_dir, 'labels')

    # Extract frames from the provided video and save them in the images directory
    extract_frames(args.path_video, dir_save=images_dir,
                   frames_per_second=2)  # Extracts and saves one frame every second

    # Load the YOLO model and annotate the extracted images
    model = YOLO(args.model_path)
    annotate_images(model, images_dir, labels_dir)

    # Convert YOLO format annotations to VOC format
    # Note: The YOLO classes file needs to be present in the specified directory
    convert_yolo_to_voc(args.save_dir, labels_dir, images_dir, args.yolo_classes)


# Check if the script is being run as the main module
if __name__ == '__main__':
    # Initialize an argument parser for command-line input
    parser = argparse.ArgumentParser(description='Process the paths.')

    # Add an argument to specify the path to the video that needs processing
    parser.add_argument('--path_video',
                        default='/Users/louis.skowronek/object-detection-project/videos/black_axle_and_black_beam.mp4')

    # Add an argument to specify the directory where results will be saved
    parser.add_argument('--save_dir', default='/Users/louis.skowronek/AISS/test_files')

    # Add an argument to specify the path to the YOLO classes file
    parser.add_argument('--yolo_classes',
                        default='/Users/louis.skowronek/object-detection-project/onnx_yolov8/classes.txt')

    # Add an argument to specify the path to the YOLO model
    parser.add_argument('--model_path', default='../models/yolov8s_best.pt')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
