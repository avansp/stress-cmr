FROM nvcr.io/nvidia/tensorflow:22.05-tf2-py3

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    build-essential \
    screen \
 && rm -rf /var/lib/apt/lists/*

# Install necessary python packages, you need
RUN python -m pip install --upgrade pip

# Install poetry
RUN pip install poetry

CMD ["bash"]