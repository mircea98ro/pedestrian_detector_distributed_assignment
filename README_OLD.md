# Distributed dynamic sensor assignment of multiple mobile targets
## Targets detected through object detection using deep learning

Given the number of targets of the problem and the ID of the robot to compute the assignment, this workspace is able to:
- The detection node obtains frames from a RGB-D camera and at each frame tries to detect pedestrians through evaluating a YOLO detector.
- Knowing the colours of each target, previously initializated, is able to establish the id of each pedestrian detected and send to the assignment node the relative cost associated to that target and its relative coordinates.
- The assignment node reads the computed costs for the robot i coming from the detection node. Then, computes the assignment with a distributed simplex assignment algorithm and with the assignment obtained for the robot i to look target j, publish the desires relatives coordinates to the PTZ node 
- The PTZ node receives those coordinates and focus towards the target using a 2 axis movement with the camera.

(**Paper:** )

This ROS workspace holds some ROS packages that enable the next features:

* _**detection**_ : Detects the targets through the camera frames, computes the assignment costs and sets the relative position of the PTZ towards each target observed. [Python]
* _**simplex**_ : Computes the assignment in a distributed way of one of the robots known the costs and send to the PTZ the coordiantes of the assignment. [C++]
* _**PTZ_ROS**_ : Known the coordinates of the target, gives commands to the servos of the PTZ towards the target assigned. It also initializes the PTZ.
* _**Camera_ROS**_ : Initializes the RGB-D camera. (Installation via librealsense2)

---

### Requirements:

* Ubuntu >=18.04
* ROS Melodic
* Python >=3.2
* OpenCV >=3.4.2 - As of *March 2022*, `pip install OpenCV` installs pre-built OpenCV __3.2.0__ binaries. To install a newer version, you need to get the source code, compile and install it. There are guides on the internet on how to do that.

### ROS Dependencies:

* eigen (Added already as a **git submodule** from [here](https://gitlab.com/libeigen/eigen). The module should be at commit `2627e2f2`. Use `git checkout 2627e2f2e6125cf09fa32789755135e84552275b`on it if is on a different commit)
*Old instructions:* [SHA hash - 70fbcf82ed0a95b27ee68e20199a4e8e1e913268]
* cv_bridge: `sudo apt-get install ros-melodic-cv-bridge`
* numpy: `pip install numpy` & `pip3 install numpy`
* dlib: `pip install dlib` & `pip3 install dlib`
* imutils: `pip install imutils` & `pip3 install imutils`
* cmake: `sudo apt-get install cmake`
* catkin: `sudo apt-get install catkin`
* librealsense2 (installation guide: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
* std_msgs `sudo apt-get install ros-melodic-std-msgs`
* usb-cam `sudo apt-get install ros-melodic-usb-cam`
* (**New**) realsense2-camera:`sudo apt-get install ros-melodic-realsense2-camera`

### YOLO detector:
The YOLO detector used in this node uses a tiny implementation of the YOLO detector to be able to compute a real time detection on CPU. 
Yolo Darknet used already implemented in OpenCV >=3.4.2.
URL of the tiny weights and parameters used (already in this repository */detection/darknet*): https://pysource.com/2019/07/08/yolo-real-time-detection-on-cpu/

---

### Installation (ROS):

0. Install ROS Melodic and source it on the setup bashrc.sh file.
1. Clone the repository into your workspace src folder and source the workspace.
2. Install the ROS dependencies needed through ROS and pip installations.
3. In a terminal open the workspace folder, and run 'catkin_make'
   1. If that returns errors, run `catkin_make simplex_generate_messages` first, then run `catkin_make` as normal
   2. If the compiler ask for some other dependencies, install it.
   3. If the compiler do not find some .h files, try running 'catkin_make -j 1' or 'catkin_make_isolated'.

4. When the compiler finishes, it's ready to use.

---

### Running:

(Previously to launch the nodes, give permissions to the PTZ USB connection: `sudo chmod 666 /dev/ttyUSB0`)

#### 0. Color initialization launcher:

_**color_initialization**_

To obtain the mean of the colours used by each target. Later, the HSV mean values associated to each target will be used to run the full launcher and to differentiate each target.

Argument | Default | Description
------------ | ------------- | -------------
N | 3 | Number of targets

Example:

**`roslaunch detection color_initialization.launch N:="3"`**

#### 1. Full launcher:

This launcher initializes the PTZ and the camera. Then, runs the detection node, the assignment node and the PTZ movement node.

_**assignment_face_detector**_

Argument | Default | Description
------------ | ------------- | -------------
robot | 1 | Id of the robot
N | 3 | Number of targets
costf | 0 | Relative cost = f(bounding box size=0) or f(central pixel depth=1)
pos_max | 1.5 | Absolute limit of the servo +-pos_max(rad)
color_select | 0 | Color identification HSV (0) or RGB (1)
colours | 101 147 50 160 207 120 66 123 69 | Colours of the targets [HSV mean] (HSV_1, HSV_2, HSV_3)

Example:

**`roslaunch detection assignment_face_detector.launch robot:="2" N:="3" costf:="0" pos_max:="1.5" color_select:="1" colours:="101 147 50 160 207 120 66 123 69"`**

#### 2. Visualization:

To visualize the image and the assignment just run **rviz** and add the topics */r_x/image* where *x* is the robot id.

#### 3. Multi-Robot System:

If, instead of just one robot with multiple targets, this launcher is going to be running in multiple robots, is needed to reconfigure the bashrc.sh file in all the robots and also in the PC where the roscore is going to be executed. To each robot and the master PC add the next lines to configure a roscore abailable to all the comptuters at the same time to be able to communicate between each other.

**Master PC:**

export ROS_MASTER_URI=http://192.168.1.XXX:11311

**Each robot:**

export ROS_MASTER_URI=http://192.168.1.XXX:11311

export ROS_HOSTNAME=192.168.1.YYY

export ROS_IP=192.168.1.YYY

*(XXX = IP of master PC // YYY = IP of each robot)*
