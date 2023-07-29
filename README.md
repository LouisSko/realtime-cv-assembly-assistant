# Object Detection Project

# Object Detection Project Setup and Use Instructions

This README provides instructions for the initial setup of the Object Detection Project and its regular use. 
It is aimed to be used with a jetson-nano developer kit running on SDK 4.6.1. However, it can be adapted to be compatible with other SDK Versions as well.
## Initial Setup

You only need to perform these steps once to set up the project.

1. **Clone the Jetson-Inference repository.** 

    Open your terminal and type the following command:

    ```
    git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
    ```

2. **Download the pre-built Docker image.** 

    For Jetpack 4.6.1, use this version. More details on other versions can be found [here](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md).

    ```
    docker pull dustynv/jetson-inference:r32.7.1
    ```

3. **Clone object-detection-project repository.** 

    ```
    git clone https://git.scc.kit.edu/aiss-applications-in-computer-vision/object-detection-project
    ```

4. **Navigate into the `object-detection-project` directory and download the correct version of onnxruntime-gpu.**
   
   More details on other versions can be found [here](https://elinux.org/Jetson_Zoo#ONNX_Runtime).

    ```
    cd object-detection-project/
    wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
    ```

## Regular Use

Perform these steps every time you want to run the application.

1. **Navigate to the `jetson-inference` directory, run the container, and mount the `object-detection-project` repository into it.** 

    ```
    cd jetson-inference/
    docker/run.sh --volume /home/aiss/object-detection-project:/object-detection-project
    ```

2. **Navigate into the `object-detection-project`.** 

    ```
    cd /object-detection-project
    ```

3. **Install onnxruntime-gpu.** 

    ```
    pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
    ```

4. **Run the application.** 

    ```
    python3 app.py
    ```

Please make sure to follow these steps in the given order. 