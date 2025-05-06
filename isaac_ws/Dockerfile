FROM nvcr.io/nvidia/isaac-sim:4.5.0


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        cmake \
        build-essential \
        zenity && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

WORKDIR /root/isaac_ws

CMD ["bash"]
