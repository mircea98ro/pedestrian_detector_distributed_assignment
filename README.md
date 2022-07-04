# Distributed dynamic sensor assignment of multiple mobile targets

Thesis [here](https://github.com/mircea98ro/mircea98ro/blob/main/Master_Thesis.pdf). Paper **coming soon**.

## Installation guide

### Pre-requisites

For old algorithm:
- **Ubuntu** [18.04](https://releases.ubuntu.com/18.04/)*(tested)* or newer *(on your own risk)*
- **ROS** [Melodic](https://wiki.ros.org/melodic/Installation/Ubuntu)
  - **Python** 2.7
- **Python** 3.2 or newer :arrow_right: `sudo apt-get install python3`

- **OpenCV** 3.4.2 :arrow_right: `pip install -I opencv==3.4.2.16` then same but with `pip3`

For new detection method:


  ❗If using also Python 2 on the same machine, use `pip3` instead of `pip` everywhere.❗
  
- **Ubuntu** [20.04](https://releases.ubuntu.com/20.04/)*(tested)* or newer *(on your own risk)*
- **ROS** [Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
- **Python** 3.8 or newer :arrow_right: `sudo apt-get install python3` *(should be included by default while installing Ubuntu/ROS Noetic)*
- **OpenCV** 3.4.2 or newer :arrow_right: `sudo apt install libopencv-dev python3-opencv`
- **PyTorch** 1.11.0 or newer :arrow_right: `pip install torch torchvision torchaudio` *(or use your preferred [way of installing](https://pytorch.org/get-started/locally/). The project is supposed to use just the CPU, no GPU)*
  - **Torchreid** v1.0.6 or newer :arrow_right: Follow instructions [here](https://github.com/KaiyangZhou/deep-person-reid), ignoring the parts with `conda`

### ROS Dependencies

- eigen (Added already as a *git submodule* from [here](https://gitlab.com/libeigen/eigen). The module should be at commit `2627e2f2`. Use `git checkout 2627e2f2e6125cf09fa32789755135e84552275b`on it if is on a different commit)
  *Original author indicates another commit as the right one:* [SHA hash - 70fbcf82ed0a95b27ee68e20199a4e8e1e913268]
- cv_bridge: `sudo apt-get install ros-noetic-cv-bridge`
- numpy: `pip install numpy` *(Skip if you already installed OpenCV)* 
- scipy: `pip install scipy`
- dlib: `pip install dlib`
- imutils: `pip install imutils`
- cmake: `sudo apt-get install cmake` *(should be installed)*
- catkin: `sudo apt-get install catkin` *(should be installed)*
- std_msgs `sudo apt-get install ros-noetic-std-msgs` *(should be installed)*
- usb-cam `sudo apt-get install ros-noetic-usb-cam` *(should be installed)*
- realsense2-camera:`sudo apt-get install ros-noetic-realsense2-camera`

### YOLO detector

The YOLO Darknet used is implemented in OpenCV 3.4.2 and newer. For the parameters used check [here](https://pysource.com/2019/07/08/yolo-real-time-detection-on-cpu/). They are also found in `{repo}/detection/darknet`.

### Installation steps

1. Install Ubuntu using preffered method.
2. Install Python 3, OpenCV, PyTorch and Torchreid as described above.❗As of May 2022, there is a problem installing with the `Pillow` package for Python. Manually install version 8.4 `pip install pillow==8.4`.
3. Install ROS Noetic, preferably the full-desktop type: `sudo apt install ros-noetic-desktop-full`. Follow the link in *pre-requisites* for more details.
4. Install ROS dependencies using the commands given above.
5. Create a catkin workspace (guide [here](https://wiki.ros.org/catkin/Tutorials/create_a_workspace)) and clone the repository into the `{workspace_path}/src` sub-folder.
6. Open a terminal in the workspace root folder and run `catkin_make`.
   1. If that returns errors, run `catkin_make simplex_generate_messages` and if that is successful, run `catkin_make` as normal afterwards.
   2. If the compiler asks for any dependencies, install them.
   3. If the compiler does not find some `.h` files, try running `catkin_make -j 1` or `catkin_make_isolated` instead of the normal command.
7. Source everything in the `.bashrc` file, run the following commands:`
   1. Source ROS: `echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc`
   2. Source the workspace: `echo "source {workspace_path}/devel/setup.bash" >> ~/.bashrc`
   3. Update current terminal session settings: `source ~/.bashrc`

### Test functionality

#### Camera functionality

Connect the camera to the PC via a USB 3 port. If working on a VM, connect the camera to the VM instead of the host OS. In a terminal, run `lsusb` and check for an Intel RealSense device.

Run `realsense-viewer`. Make sure that the camera is detected, turn on the <u>Stereo Module</u> and the <u>RGB Camera</u> and if everything displays right, the camera itself is working properly.

To test if it works with ROS:

- start the OS (run `roscore` and leave it in the background)
- start a node that captures camera data: `roslaunch realsense2_camera rs_camera.launch`
- run the visualizer tool: `rviz`
- In the visualizer tool
  -  in the left panel set the *Fixed Frame* to `camera_depth_frame`
  - Click on the lower left button `Add - By topic - /camera/color/image_raw - Image` and click OK
  - Repeat and add `/camera/depth/image_rect_raw - DepthCloud` and click OK

If rviz shows both the image and the 3D projection, then everything is working fine.

#### Detection package

- Launch a camera node using `roslaunch camera_node camera_node`, or play a pre-recorded rosbag file.
- Launch a terminal in the `detection_osnet/src/` subfolder
- Use the terminal to launch the robot nodes `roslaunch detection_osnet robot_nodes.launch generate_configure:=true level:=kalman`
- Use a terminal to launch a master node `rosrun detection_osnet master.py`. The node prints the overall statistics upon termination.
- Use rviz/rqt for visualizing the resulted images. Use your favorite tool for visualizing the statistics (recommended [PlotJuggler](https://github.com/PlotJuggler/plotjuggler-ros-plugins))
