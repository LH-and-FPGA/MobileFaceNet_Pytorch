FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 构建工具 + ARM 交叉编译工具链
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    gcc-arm-linux-gnueabihf \
    g++-arm-linux-gnueabihf \
    crossbuild-essential-armhf \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 升级 CMake（onnxruntime 需要 >= 3.26）
RUN pip3 install cmake --upgrade

# Python 构建依赖
RUN pip3 install numpy packaging wheel setuptools flatbuffers sympy

# ARM 版 Python 头文件（交叉编译 pybind11 需要）
RUN mkdir -p /usr/include/arm-linux-gnueabihf/python3.10 && \
    cp /usr/include/python3.10/pyconfig.h /usr/include/arm-linux-gnueabihf/python3.10/pyconfig.h

WORKDIR /workspace
