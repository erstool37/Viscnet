FROM nvidia/opencl:latest

RUN apt update && \
    apt install openssh-server openssh-client python3 curl git build-essential vim -y && \
    apt install gcc-8 g++-8 -y && \
    update-alternatives —install /usr/bin/gcc gcc /usr/bin/gcc-8 800 —slave /usr/bin/g++ g++ /usr/bin/g++-8