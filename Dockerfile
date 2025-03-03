FROM nvidia/cuda@sha256:b3e7fba84d169f46939f00c25be7d016f712a8d651f4756d6a55e693d84d94f2
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WORKDIR /usr/local/app

RUN apt-get update -q \
    && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY . ./llmrl
RUN ln -s /usr/bin/python3.11 /usr/bin/python
#RUN cd llmrl && python3.11 -m pip install -r requirements.txt
RUN cd llmrl && make develop
