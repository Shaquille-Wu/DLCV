FROM ubuntu:16.04

## for fastblas armv7/v8_linux
RUN apt-get update -y \
    && apt-get install -y gcc-arm-linux-gnueabihf gcc-aarch64-linux-gnu
## for fastblas x86_linux
RUN apt-get update -y \
    && apt-get install -y gcc git python

## for iml-score build x86_linux
RUN apt-get update -y \
    && apt-get install -y g++ cmake vim
## for iml-score build armv7/v8_linux
RUN apt-get update -y \
    && apt-get install -y g++-arm-linux-gnueabihf g++-aarch64-linux-gnu

# cmake
ADD https://github.com/Kitware/CMake/releases/download/v3.16.0-rc3/cmake-3.16.0-rc3-Linux-x86_64.tar.gz /
WORKDIR /
RUN tar -xzf cmake-3.16.0-rc3-Linux-x86_64.tar.gz
ENV PATH="/cmake-3.16.0-rc3-Linux-x86_64/bin:${PATH}"

RUN apt-get update -y \
    && apt-get install -y libopencv-dev
#&& apt-get install -y libprotobuf-dev protobuf-compiler

