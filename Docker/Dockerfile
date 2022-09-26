ARG CUDA_VERSION=11.2.2
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=20.04
# pick version based on your host NVIDIA driver version:
# see Table 3 at https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions
# - 10.1-cudnn7-devel-ubuntu18.04
# - 11.4.1-devel-ubuntu20.04
# - 11.2.2-cudnn8-devel-ubuntu20.04
# - 11.7.1-cudnn8-devel-ubunty20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}


ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.5.3

RUN apt-get update && apt-get upgrade -y && \
    # Install build tools, build dependencies and python
    apt-get install -y \
        build-essential \
        cmake \
        git \
        nano \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        libpcl-dev \
        libfmt-dev \
        ## Python
        python3-pip \
        python3-dev \
        python3-numpy
#    && rm -rf /var/lib/apt/lists/*

# Install Ceres library
# NOTE: this is installed by the Sophys dependencies script
# http://ceres-solver.org/installation.html#linux
# RUN apt-get install -y -qq --no-install-recommends \
#     libgoogle-glog-dev libgflags-dev libatlas-base-dev \
#     libeigen3-dev libsuitesparse-dev
# RUN git clone -b 2.1.0 https://github.com/ceres-solver/ceres-solver.git && \
#     cd ceres-solver && mkdir build && cd build && \
#     cmake .. && \
#     make -j"$(nproc)" && \
#     # Install to /usr/local/lib
#     make install

# Sophus dependency script from https://github.com/strasdat/Sophus/blob/v22.04.1/scripts/install_linux_deps.sh
RUN apt-get install -y -qq --no-install-recommends \
    gfortran libc++-dev libgoogle-glog-dev libatlas-base-dev \
    libsuitesparse-dev libceres-dev ccache && \
    # Eigen library
    cd /opt && \
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.tar.bz2 && \
    tar xvf eigen-3.3.4.tar.bz2 && \
    mkdir build-eigen && \
    cd build-eigen && \
    cmake ../eigen-3.3.4 -DEIGEN_DEFAULT_TO_ROW_MAJOR=$ROW_MAJOR_DEFAULT && \
    make install && \
    # Ceres library
    cd /opt && \
    git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver && \
    cd ceres-solver && \
    git reset --hard 399cda773035d99eaf1f4a129a666b3c4df9d1b1 && \
    mkdir build && \
    cd build && \
    ccache -s && \
    cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache .. && \
    make -j"$(nproc)" && \
    make install

# Install Sophus Library
RUN cd /opt && \
    git clone -b v22.04.1 https://github.com/strasdat/Sophus.git && \
    cd Sophus && mkdir build && cd build && \
    cmake .. && \
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install


# Install g2o Library
RUN cd /opt && \
    git clone -b 20201223_git https://github.com/RainerKuemmerle/g2o.git && \
    cd g2o && mkdir build && cd build && \
    cmake .. && \
    make -j"$(nproc)" && \
    make install

# Install OpenCV
RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}