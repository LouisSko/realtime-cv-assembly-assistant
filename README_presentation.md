# Object Detection Project

# Object Detection Project Setup and Use Instructions

This README provides instructions for the initial setup of the Object Detection Project and its regular use. 
It is aimed to be used with a jetson-nano developer kit running on SDK [4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461). However, due to the implementation choices, it can be easily adapted to be compatible with other SDK Versions as well.
Simply use a compatible pre-built Jetson-Inference docker image and download the compatible onnxruntime-gpu version.

## Initial Setup

You only need to perform these steps once to set up the project.

1. **Clone object-detection-project repository.** 

    ```
    git clone https://git.scc.kit.edu/aiss-applications-in-computer-vision/object-detection-project
    ```
   
2. **Download the pre-built Docker image Jetson-Inference.** 

    For Jetpack 4.6.1, use this version. More details on other versions can be found [here](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md).

    ```
    docker pull dustynv/jetson-inference:r32.7.1
    ```

3. **Navigate into the `object-detection-project/downloads` directory and download the correct version of onnxruntime-gpu.**
   
   For Jetpack 4.6.1, use this version. More details on other versions can be found [here](https://elinux.org/Jetson_Zoo#ONNX_Runtime).

    ```
    cd path/to/object-detection-project/downloads
    ```
    ```
    wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
    ```

## Regular Use

Perform these steps every time you want to run the application.


1. **Navigate into the `object-detection-project` and start the container.** 

    ```
    cd path/to/object-detection-project/docker/run.sh
    ```

2. **Install onnxruntime-gpu.** 

    ```
    pip3 install downloads/onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
    ```

3. **Run the Object Detection Script.** 

   ```
   python3 inference/app.py --model_name yolov8s_best.onnx --use_camera_stream --skip_frames 5
   ```


Please make sure to follow these steps in the given order. 



## Enhancing Inference Performance on Jetson Nano Developer Kit

To optimize the performance of the inference on the Jetson Nano Developer Kit, we can switch to a lightweight desktop environment. This reduces memory usage substantially - in our case from 2 GB to approximately 500 MB. 

Follow these steps to switch to the Lightweight X11 Desktop Environment (LXDE):

1. **Select LXDE as a desktop environment.**

    Instead of using the default "Unity" desktop environment, we will use the LXDE, which is more lightweight and uses less system resources.

2. **Switch GNOME display manager (gdm3) to lightdm.**

    The `lightdm` display manager is more lightweight and uses less system resources than `gdm3`. Switch to it by entering the following command in your terminal:

    ```
    sudo dpkg-reconfigure lightdm
    ```

3. **Reboot the Jetson Nano.**

    After changing your desktop environment and display manager, reboot your Jetson Nano to apply these changes.

For more information and detailed instructions, check out this tutorial: [Save 1GB of Memory – Use LXDE on your Jetson](https://jetsonhacks.com/2020/11/07/save-1gb-of-memory-use-lxde-on-your-jetson/).



# Data Gathering and Training

Following scripts are not supposed to be executed on the Jetson Nano.


## Automated Image Generation Pipeline

The script image_generation.py processes a video file and uses a trained YOLOv8 model for object detection on each frame. The detected objects are then annotated in the images, and the YOLO-formatted annotations are converted to VOC format. This is useful for generating annotated image datasets from video footage.


### Usage

The script can be run from the command line and takes four optional arguments:

1. `--path_video`: The path to the video file to process. Default is `/Users/louis.skowronek/object-detection-project/videos/black_axle_and_black_beam.mp4`.
2. `--save_dir`: The directory where the processed images and labels will be saved. Default is `/Users/louis.skowronek/AISS/test_files`.
3. `--yolo_classes`: The path to the .txt file containing the classes for the YOLO model. Default is `/Users/louis.skowronek/object-detection-project/onnx_yolov8/classes.txt`.
4. `--model_path`: The path to the trained YOLO model. Default is `../models/yolov8s_best.pt`.

To run the script with all paths specified:

```
python image_generation.py --path_video /path/to/video --save_dir /path/to/save/dir --yolo_classes /path/to/yolo/classes --model_path /path/to/model
```



## Splitting the Dataset into Train, Test, and Validation Sets

The Python script `split_dataset.py` allows you to split a dataset into training, testing, and validation sets according to a specified ratio. The dataset should contain 'images' and 'labels' folders.

### Usage

The script can be run from the command line and takes three optional arguments:

1. `--input_dir`: Specifies the path to the directory containing the dataset. The script expects this directory to contain 'images' and 'labels' folders. Default is `/Users/louis.skowronek/AISS/generate_images`.

2. `--output_dir`: Specifies the path to the output directory where the split datasets (training, testing, validation) will be saved. If this argument is not specified, the input directory will be used as the output directory. Default is the same as `--input_dir`.

3. `--split_ratio`: Specifies a tuple of three values representing the proportion of data to be used for training, testing, and validation, respectively. The values should add up to 1. Default is `0.8,0.1,0.1` (representing 80% training, 10% testing, and 10% validation)

To run the script with all arguments specified:

```
python split_dataset.py --input_dir /path/to/input --output_dir /path/to/output --split_ratio 0.8,0.1,0.1
```

## Image Augmentation

This script augments images from a provided directory and stores the augmented images in a specified output directory.
Note: Using ultralytics to train a YOLOV8 has a built in image augmentation function

### Usage

The script can be run from the command line and takes three optional arguments:

1. `--input_dir`: The directory containing the images to be augmented. Default is `/Users/louis.skowronek/AISS/aiss_images/train`.
2. `--output_dir`: The directory where the augmented images will be saved.  If this argument is not provided, the script will use the input directory as the output directory.
3. `--nr_of_augs`: The number of augmented images to be generated per original image. Default is 10.

To run the script with all arguments specified:

```
python image_augmentation.py --input_dir /path/to/input --output_dir /path/to/output --nr_of_augs 5
```

## Image Resizing

This script is used to resize images in directories. It takes as input the base directory which contains subdirectories ('train', 'val', 'test') and each subdirectory should have an 'images' folder with the images to be resized.

The script can be run from the command line with the following arguments:

1. `--input_dir_base` : The base input directory path. This directory should contain subdirectories ('train', 'val', 'test') and each subdirectory should have an 'images' folder with images to be resized. Default is `/Users/louis.skowronek/AISS/generate_images/yolo_images.
2. `--output_dir_base` : The base output directory path. If this argument is not provided, the script will use the input directory as the output directory.
3. `--max_width` : The maximum width for the resized images. Default is 1280.
4. `--max_height` : The maximum height for the resized images. Default is 720.

To run the script with all arguments specified:

```
python resize_images.py --input_dir_base /path/to/input/base --output_dir_base /path/to/output/base --max_width 1280 --max_height 720
```



# Unit tests

unit tests are provided to test basic functionality of the http requests.

run tests using:

```
pytest --cov-report term --cov=inference tests/
```

