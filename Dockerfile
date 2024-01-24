FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C TZ=UTC TERM=linux

# install some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    wget \
    python3-dev \
    python3-pip \
    python3-tk\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt ;
