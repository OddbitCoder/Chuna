FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Europe/Ljubljana
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

# install utils

RUN apt-get install ffmpeg -y
RUN apt-get install git -y

# install Python

RUN apt-get install python3 -y
RUN apt-get install python3-pip -y

# install Deepface

RUN mkdir /app
WORKDIR /app

RUN git clone https://github.com/serengil/deepface.git .
RUN pip install -e .

# enable GPU  

RUN pip install tensorflow-gpu==2.11.*

RUN pip install tensorrt
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/tensorrt_libs:${LD_LIBRARY_PATH}"
RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.8 /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.7
RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer_plugin.so.8 /usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer_plugin.so.7

# this is another way that I found

#RUN pip install --upgrade setuptools pip
#RUN pip install nvidia-pyindex
#RUN pip install nvidia-tensorrt==8.4.0.6
#ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/tensorrt:${LD_LIBRARY_PATH}"
#RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt/libnvinfer.so.8 /usr/local/lib/python3.10/dist-packages/tensorrt/libnvinfer.so.7
#RUN ln -s /usr/local/lib/python3.10/dist-packages/tensorrt/libnvinfer_plugin.so.8 /usr/local/lib/python3.10/dist-packages/tensorrt/libnvinfer_plugin.so.7

# These packages are optional in Deepface. Activate if your task depends on one.

RUN pip install mediapipe # required for mediapipe detector
RUN pip install ultralytics # required for yolov8 detector

RUN pip install dlib==19.20.0
#RUN pip install cmake==3.24.1.1
#RUN pip install lightgbm==2.3.1

# gRPC stuff

RUN pip install grpcio
RUN pip install grpcio-tools
RUN pip install protobuf==3.20.*

# all done

ENTRYPOINT ["python3"]
CMD ["/files/server.py"]
