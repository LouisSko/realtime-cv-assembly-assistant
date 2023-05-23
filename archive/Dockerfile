# PyTorch
FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3

# Here is a Base Image from Tensorflow (not needed for this project)
#FROM nvcr.io/nvidia/l4t-tensorflow:r34.1.0-tf2.8-py3

#Required for libjasper-dev
RUN echo "deb http://ports.ubuntu.com/ubuntu-ports/ xenial-security main restricted" | tee -a /etc/apt/sources.list

# install cmake
RUN apt-get install cmake -y libprotobuf-dev protobuf-compiler && \
	pip3 install Cython && \
    pip3 install onnx==1.10

#INSTALL OPENCV dependencies
RUN apt update && apt purge *libopencv* && apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    python2.7-dev python3.6-dev python-dev python-numpy python3-numpy \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
    libv4l-dev v4l-utils qv4l2 v4l2ucp \
    curl unzip && \
    rm -rf /var/lib/apt/lists/*

#GET OPENCV sources
WORKDIR /usr/local/src
RUN curl -L https://github.com/opencv/opencv/archive/4.1.0.zip -o opencv-4.1.0.zip && \
    curl -L https://github.com/opencv/opencv_contrib/archive/4.1.0.zip -o opencv_contrib-4.1.0.zip && \
    unzip opencv-4.1.0.zip && \
    unzip opencv_contrib-4.1.0.zip && \
    rm -rf opencv*.zip \

#INSTALL OPENCV
RUN cd opencv-4.1.0/ && mkdir release && cd release/ && \
    cmake -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_PC_FILE_NAME=opencv.pc -D WITH_CUDA=ON -D CUDA_ARCH_BIN="5.3" -D CUDA_ARCH_PTX="" -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j3 && \
    make install && \
    cp unix-install/opencv.pc /usr/local/lib/pkgconfig && \
    rm -rf /usr/local/src/opencv-4.1.0


# install pip
RUN apt-get install python3-pip
RUN pip3 install --upgrade pip

# install git
RUN apt update
RUN apt install git-all -y

# install the nano editor
RUN apt update
RUN apt install nano

# install the jetcam 
RUN git clone https://github.com/NVIDIA-AI-IOT/jetcam.git && \
    cd jetcam && \
    git checkout 508ff3a && \
    python3 setup.py install && \
    cd ../ && \
    rm -rf jetcam
 
RUN pip3 install traitlets

# install dlib
RUN apt-get install libjpeg-dev 
RUN wget http://dlib.net/files/dlib-19.21.tar.bz2 
RUN tar jxvf dlib-19.21.tar.bz2 && \
	cd dlib-19.21/ && \
	mkdir build && \
	cd build/&& \
	cmake .. && \
	cmake --build . && \
	cd ../ && \
	python3 setup.py install 


# install torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
	cd torch2trt && \ 
	python3 setup.py install

# requirements
RUN pip3 install scipy
RUN pip3 install imutils
RUN pip3 install requests
RUN pip3 install tqdm
RUN pip3 install PyYAML
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install seaborn
RUN pip3 install inputimeout
RUN pip3 install thop
RUN pip3 install Flask




