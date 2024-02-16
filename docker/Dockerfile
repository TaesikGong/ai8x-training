FROM nvcr.io/nvidia/pytorch:21.10-py3

ARG USER
ARG UID
ARG GID
ARG DEBIAN_FRONTEND=noninteractive

# personal apps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    bzip2 \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    software-properties-common \
    unzip \
    zip \
    sudo \
    screen \
    tmux

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${USER}" | chpasswd && adduser ${USER} sudo
USER ${UID}:${GID}
WORKDIR /home/${USER}

# required apps
RUN sudo apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libsndfile-dev \
    portaudio19-dev \

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /home/${USER}/miniconda && \
    rm miniconda.sh
ENV PATH /home/${USER}/miniconda/bin:$PATH
RUN conda update -y conda
RUN conda create -y --name ai8x-training python=3.9
SHELL ["conda", "run", "-n", "project", "/bin/bash", "-c"]


# personal packages
RUN pip install --no-cache-dir -U pip && \
    pip install gpustat && \
    pip install matplotlib && \
    pip install tqdm && \
    pip install pandas

# required packages
RUN pip3 install -U pip wheel setuptools && \
    pip3 install -r requirements-cu11.txt

# Set the default command for the container (this can be changed to whatever command you need)
CMD ["conda", "run", "-n", "project", "python", "-c", "print('Conda environment setup with project requirements is ready!')"]