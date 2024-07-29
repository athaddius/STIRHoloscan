# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


############################################################
# Base image
############################################################

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v2.1.0-dgpu
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# For benchmarking
RUN apt update \
    && apt install --no-install-recommends -y \
    libcairo2-dev \
    libgirepository1.0-dev \
    gobject-introspection \
    libgtk-3-dev \
    libcanberra-gtk-module \
    graphviz

COPY benchmarks/holoscan_flow_benchmarking/requirements.txt /tmp/benchmarking_requirements.txt
RUN pip install -r /tmp/benchmarking_requirements.txt
RUN pip install scipy opencv-python opencv-contrib-python tqdm
RUN cd /tmp/ && git clone https://github.com/athaddius/MFT_STIR.git \
    && cd MFT_STIR && pip install .

# For STIRLoader
RUN mkdir -p /tmp \
    && cd /tmp/ && git clone https://github.com/athaddius/STIRLoader.git \
    && cd STIRLoader && pip install .
RUN pip install torchvision onnxruntime-gpu
RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6
