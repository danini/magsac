FROM python:3.8

RUN apt-get update \
    && apt-get install -y \
        cmake \
        libeigen3-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libopencv-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt \
    && git clone https://github.com/davidstutz/glog \
    && cd /opt/glog \
    && cmake -H. -Bbuild -G "Unix Makefiles" \
    && cmake --build build

RUN cd /opt \
    && git clone https://github.com/danini/magsac --recursive \
    && cd /opt/magsac \
    && CMAKE_PREFIX_PATH=/opt/glog/build pip install . \
    && python -c "import pymagsac"
