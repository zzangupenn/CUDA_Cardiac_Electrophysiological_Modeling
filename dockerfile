FROM nvidia/cudagl:11.4.2-devel
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install build-essential cmake libglfw3 libglfw3-dev libglew-dev
COPY . /cardiac 
RUN mkdir build
WORKDIR /cardiac/build
RUN cmake ..
RUN make -j4
COPY ./data /cardiac/build/bin/data
COPY ./shaders /cardiac/build/bin/shaders
WORKDIR /cardiac/build/bin



