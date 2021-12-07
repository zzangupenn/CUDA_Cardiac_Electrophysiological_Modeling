# CUDA_Cardiac_Electrophysiological_Modeling

**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Final Project**

* Zirui Zang
  * [LinkedIn](https://www.linkedin.com/in/zirui-zang/)
* Tested on: Windows 10 and Ubuntu 20.04, AMD Ryzen 7 3700X @ 3.60GHz 32GB, RTX2070 SUPER 8GB (Personal)
* This project is closed source until publication.

## Mitchell-Schaeffer Model Simulation
<p align="center">
<img src="images/two_point.gif"
     alt="two_point"
     width="700"/>
</p>

## Running the Code
### On Windows or Linux
Please use the dockerfile for compiling the code. Install docker and nvidia-docker2.
Then build the docker image called cardiac. This will compile the code inside the docker image.
```
docker build . -t cardiac
```
Run the simulation:
```
docker run cardiac --gpus 0 ./cuda_cardiac 
```
You change the `sim_inputs.json` or `sim_settings.json` file in the data folder. Please run the build command again to copy the files into the image.

### On NVIDIA Jetson
This repo is tested on Jetson Xaiver NX with Jetpack 4.6. Please follow this [link](https://elinux.org/Jetson/Installing_ArrayFire#GLFW) to install the GLFW on Jetson.

Then run the following in the repo folder:
```
mkdir build
cd build
cmake ..
make -j4
cp ../shaders /bin/shaders
cp ../data /bin/data
cd bin
./cuda_cardiac
```
On Jetson Xaiver NX, this code is tested with 20W4Core option.
