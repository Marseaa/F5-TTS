FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-11.8 \
    TORCH_CUDA_ARCH_LIST="8.6"

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y wget

# Instalando Python e pip
RUN apt-get install -y python3  && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y ffmpeg
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
#RUN pip install git+https://github.com/SWivid/F5-TTS.git
RUN pip install replicate

RUN pip install --upgrade transformers torch soundfile librosa

#RUN pip install gradio git+https://github.com/SWivid/F5-TTS.git
RUN pip install gradio

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /workspace

RUN git clone https://github.com/SWivid/F5-TTS.git \
    && cd F5-TTS \
    && git submodule update --init --recursive \
    && sed -i '7iimport sys\nsys.path.append(os.path.dirname(os.path.abspath(__file__)))' src/third_party/BigVGAN/bigvgan.py \
    && pip install -e . --no-cache-dir

RUN pip install appPublic
RUN pip install statsmodels

ENV PYTHONPATH $PYTHONPATH:/app/F5TTS

ENV NVIDIA_VISIBLE_DEVICES all
WORKDIR /app

CMD ["python3", "test-F5TTS.py"]
