FROM tensorflow/tensorflow:latest-gpu-py3


RUN apt-get update -y --fix-missing

RUN apt-get install -y ffmpeg

RUN apt-get install -y build-essential cmake pkg-config \
                    libjpeg8-dev libtiff5-dev \
                    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
                    libxvidcore-dev libx264-dev \
                    libgtk-3-dev \
                    libatlas-base-dev gfortran \
                    libboost-all-dev


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y wget vim python3-tk python3-pip

WORKDIR /

ADD $PWD/requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

RUN pip3 install dlib

EXPOSE 5000

CMD ["/bin/bash"]