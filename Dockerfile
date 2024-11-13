FROM python:3.12

#COPY sources.list /etc/apt/sources.list

# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 871923D1991BC93C

# RUN apt-get update && apt-get install -y software-properties-common

ARG DEBIAN_FRONTEND=noninteractive

# RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && apt-get -y install \
    python3-pip \
    libglib2.0-dev \
    libpng-dev \
    libtool \
    libx11-dev \
    tcsh \
    libbz2-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget && \
    apt-get clean all && \
    rm -r /var/lib/apt/lists/*

RUN pip3 install -i https://pypi.mirrors.ustc.edu.cn/simple/ numpy \
    torch \
    torchvision \
    Pillow \
    scikit-image \
    tqdm \
    opencv-python \
    astropy \
    numba \
    matplotlib \
    minio \
    scipy \
    pyyaml \
    boto3 \
    pymongo \
    nnunetv2

COPY . /home/soft/FRBSlopeDetection

WORKDIR /home/soft/FRBSlopeDetection