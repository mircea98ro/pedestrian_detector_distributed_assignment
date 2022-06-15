#!/usr/bin/env python3

# Robot (slave) node. Can read raw images and analyse them using YOLO & OSNET.

# TFM - Mircea-Alexandru Popovici
# Directors - Eduardo Montijano, Danilo Tardioli, Adrian Burlacu




#   Imports

import sys  # Standard system tools
from collections import deque

import cv2  # OpenCV for Image processing
import imutils  # Image processing utility package
import numpy  # Math library
import rospy  # ROS-Python Interface
from cv_bridge import CvBridge  # Interface between OpenCV and ROS imaging
from detection_osnet.msg import (  # Custom message types between robot and monitor
    ProcessData, ProcessWindow, Window)
from scipy.optimize import \
    linear_sum_assignment  # Hungarian algorithm (bipartite assignment)
from scipy.spatial import distance  # Distance measuring for vectors
from sensor_msgs.msg import Image  # ROS standard image message data class

from reid import ReID  # Torch Re-ID class
from yolo import YOLO  # YOLO class

#   Defines

DETECT_LVL_RAW = 0
DETECT_LVL_YOLO = 1
DETECT_LVL_OSNET = 2

#   Globals

bridge = CvBridge()
yolo : YOLO = None
reid : ReID = None
robot = None

#   Classes

#   Main

class Robot:

    def __init__(self):
        self.pending = False

        self.imgS = rospy.Subscriber('r_1/camera/color/image_raw', Image, self.callback, queue_size=1)
        self.dataP = rospy.Publisher('r_1/data', ProcessData, queue_size=1)

        self.img_raw = None
        self.img_cv2 = None
        self.msg = ProcessData()

        try:
            if (type(sys.argv[1]) is int):
                self.detection_level = sys.argv[1]
        except Exception:
            self.detection_level = DETECT_LVL_YOLO

        try:
            self.dynamic_gallery = sys.argv[2]
        except Exception:
            self.dynamic_gallery = True

        try:
            self.dynamic_gallery_content = sys.argv[2]
        except Exception:
            self.dynamic_gallery_content = False
            
        
        if (type(self.dynamic_gallery) is bool):
            if (self.dynamic_gallery == True):
                self.feature_galleries = deque()

    def callback(self, msg):
        if self.pending == True:
            return
        self.img_raw = msg
        self.pending = True
        # img_cv2 = bridge.imgmsg_to_cv2(self.image_raw, "bgr8")
        return self

    def detect(self):
        global yolo
        
        start_time = rospy.Time.now()
        self.msg.frame_seq = self.img_raw.header.seq
        self.img_cv2 = bridge.imgmsg_to_cv2(self.img_raw, "bgr8")

        height, width, channels = self.img_cv2.shape
        
        self.msg.windows = []

        if (self.detection_level > DETECT_LVL_RAW):
            # YOLO Detection
            blob = cv2.dnn.blobFromImage(self.img_cv2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            yolo.network.net.setInput(blob)
            outs = yolo.network.net.forward(yolo.network.output_layers)

            boxes = []
            for out in outs:
                for detection in out:
                    if (detection[4] < 0.8): # TODO: Make variable confidence score
                        continue
                    scores = detection[5:]
                    class_id = numpy.argmax(scores)
                    score = scores[class_id]
                    if (score > yolo.params.min_score) and (class_id == 0):
                        # Object detected
                        x = int(detection[0] * width)
                        y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Register data
                        boxes.append(Window(x = x, y = y, w = w, h = h))
            # Filter data
            for i in yolo.filter_windows(boxes):
                self.msg.windows.append(ProcessWindow(yolo = boxes[i], osnet = None, assignment = None))

        if (self.detection_level > DETECT_LVL_YOLO):
            # OSNET Detection
            
            window_scores = []

            for pw in self.msg.windows:
                # Consider desired aspect ratio 1:2 (w:h)
                diff = pw.yolo.h - pw.yolo.w*reid.params.window_ratio
                if diff > 0:
                    pw.osnet.h = pw.yolo.h
                    pw.osnet.w = int(pw.yolo.h / reid.params.window_ratio)
                elif diff < 0:
                    pw.osnet.h = int(pw.yolo.w * reid.params.window_ratio)
                    pw.osnet.w = pw.yolo.w
                else:
                    pw.osnet.h = pw.yolo.h
                    pw.osnet.w = pw.yolo.w
                pw.osnet.x = pw.yolo.x
                pw.osnet.y = pw.yolo.y
            
            descriptor = []
            try:
                for win in self.msg.windows:
                    xLeft = int(max(0, win.osnet.x - win.osnet.w/2))
                    yUp = int(max(0, win.osnet.y - win.osnet.h/2))
                    xRight = int(min(width, win.osnet.x + win.osnet.w/2 - 1))
                    yDown = int(min(height, win.osnet.y + win.osnet.h/2 - 1))

                    cropped = self.img_cv2[xLeft:xRight, yUp:yDown]
                    cropped = imutils.resize(cropped, width = reid.params.window_w, height = int(reid.params.window_w*reid.params.window_ratio))
                    descriptor.append(reid.extractor(cropped))
                    gallery_score = []
                    for gallery in self.feature_galleries:
                        scores = []
                        for element in gallery:
                            scores.append(distance.cosine(descriptor[len(descriptor)-1], element))
                        gallery_score.append(numpy.min(scores))
                    window_scores.append(gallery_score)

                window_scores = numpy.matrix(window_scores)
                
                row_index, col_index = linear_sum_assignment(window_scores)

                for i in range(len(row_index)):
                    if window_scores[row_index[i], col_index[i]] > reid.params.max_distance:
                        continue
                    self.msg.windows[row_index[i]].assignment = col_index[i] + 1

                    if self.dynamic_gallery_content:
                        if len(self.feature_galleries[col_index[i]]) >= reid.params.descriptor_size:
                            self.feature_galleries[col_index[i]].popleft()
                    if len(self.feature_galleries[col_index[i]]) < reid.params.descriptor_size:
                        self.feature_galleries[col_index[i]].append(descriptor[row_index[i]])
                
                if self.dynamic_gallery:
                    for index in range(len(self.msg.windows)):
                        if (self.msg.windows[index].assignment == 0):
                            gal = deque()
                            gal.append(descriptor[index])
                            self.feature_galleries.append(gal)
                            self.msg.windows[index].assignment = len(self.feature_galleries)
                            
            except Exception as e:
                print(str(e))
                rospy.logerr(f'Frame #{self.msg.header.seq} had windows out of bounds and was processed!')
                
        self.msg.delta = rospy.Time.now() - start_time
        self.msg.header.stamp = rospy.Time.now()
        self.dataP.publish(self.msg)
        self.pending = False
        # rospy.logdebug("Robot: Detection occured!")

#   Functions

def robot_detection():
    global yolo
    global reid

    files = YOLO.YOLOFiles(rospy.get_param("/cfg/yolo_path"), rospy.get_param("/cfg/yolo_weight"), rospy.get_param("/cfg/yolo_cfg"), rospy.get_param("/cfg/yolo_names"))
    params = YOLO.YOLOParameters(rospy.get_param("/cfg/yolo_min_score"), rospy.get_param("/cfg/yolo_max_overlap"), rospy.get_param("/cfg/yolo_max_count"))
    yolo = YOLO(params = params, files = files)

    files = ReID.ReIDFiles(rospy.get_param("/cfg/osnet_path"), rospy.get_param("/cfg/osnet_model"))
    params = ReID.ReIDParameters(rospy.get_param("/cfg/osnet_descriptor_size"), rospy.get_param("/cfg/osnet_window_w"), rospy.get_param("/cfg/osnet_window_ratio"), rospy.get_param("/cfg/osnet_max_distance"))
    reid = ReID(files, params)

    robot = Robot()

    rospy.loginfo("Robot node initialized succesfully. Detection started.")

    while not rospy.is_shutdown():
        if (robot.pending == True):
            robot.detect()

#   Guard
if __name__ == '__main__':    
    
    # Initialize node
    rospy.init_node("robot", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Robot node initializing!")

    # Launch in execution
    try:
        robot_detection()
    except rospy.ROSInterruptException:
        rospy.logerr("Robot node initialization error!")
        pass
