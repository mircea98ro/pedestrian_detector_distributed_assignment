#!/usr/bin/env python3

# TARGETS CAMERA DETECTOR NODE - target_tracking (no Args)
# Use YOLO for detection, OSNET for re-identification

# TFM - Mircea-Alexandru Popovici
# Directors - Eduardo Montijano, Danilo Tardioli, Adrian Burlacu

#   Imports
from dataclasses import dataclass   # Structs
from collections import deque
from matplotlib.pyplot import box

from soupsieve import match
from torch import Tensor       # Queues            
import rospy                        # ROS - Python interface
import sys                          # Standard system tools
import cv2                          # OpenCV
import time
import math
import os
import numpy as np  # Math engine

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Int32, Int32MultiArray
from torchreid.utils import FeatureExtractor
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

import argparse
import imutils

#   Init declarations

# OpenCV Bridge
bridge = CvBridge()

# YOLO

yolo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'detection', 'darknet')
weight_path = os.path.realpath(os.path.join(yolo_path,'yolov3-tiny.weights'))
# weight_path = "/home/mircea98ro/catkin_ws/src/pedestrian_detector_distributed_assignment/detection/darknet/yolov3-tiny.weights"
tiny_path = os.path.realpath(os.path.join(yolo_path,'yolov3-tiny.cfg'))
# tiny_path = "/home/mircea98ro/catkin_ws/src/pedestrian_detector_distributed_assignment/detection/darknet/yolov3-tiny.cfg"
names_path = os.path.realpath(os.path.join(yolo_path,'coco.names'))
# names_path = "/home/mircea98ro/catkin_ws/src/pedestrian_detector_distributed_assignment/detection/darknet/coco.names"

net = cv2.dnn.readNet(weight_path, tiny_path)
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# OSNET

osnet_path = os.path.join(os.path.dirname(__file__), '..', 'osnet_models')
osnet_model_name = 'osnet_ain_x0_25' 
osnet_model_path = os.path.join(osnet_path, osnet_model_name + '_imagenet.pyth')

# Others

colors = np.random.uniform(0, 255, size=(10, 3))
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
start_timer = rospy.Time(0.0)

#   Functional code

# Classes

class Frame:

    extractor = FeatureExtractor(model_name=osnet_model_name, model_path=osnet_model_path, device='cpu')

    @dataclass
    class box:
        x: float
        y: float
        w: float
        h: float
        a: float
        c_x: float
        c_y: float
    
    @dataclass
    class param_data:
        min_score: float
        max_overlap: float
        max_count: int
    
    @dataclass
    class timestamped_descriptor:
        descriptor : Tensor
        timestamp: rospy.Time

    @dataclass
    class gallery_with_history:
        gallery : deque
        assignment : int


    def __init__(self, min_score = 0.6, max_overlap = 0.3):
        # Initialize ROS Publisher(s) & Subscriber(s)
        self.subscriber = rospy.Subscriber(
            'r_1/camera/color/image_raw', Image, self.callback, queue_size=1)
        self.publisher = rospy.Publisher(
            'tracking/target_tracking', Image, queue_size=1)
        self.logs = rospy.Publisher(
            'tracking/target_tracking_log', String, queue_size=1)
        self.image_raw = None
        self.image_cv2 = deque()
        self.image_process = None
        self.yolo_windows = []
        self.osnet_windows = []
        self.assignments = []
        self.image_yolo = None
        self.image_pre_osnet = None
        self.image_osnet = None
        self.tune = self.param_data(min_score, max_overlap, 5)
        self.features_gallery = []
        self.rcv_times = deque()
        self.rcv_frame_count = 0
        self.process_rcv_time = 0.0
        self.process_frame_count = 0
        return

    def callback(self, msg):
        if (len(self.image_cv2) > self.tune.max_count):
            self.image_cv2.popleft()
            self.rcv_times.popleft()
        self.image_raw = msg
        self.image_cv2.append(bridge.imgmsg_to_cv2(self.image_raw, "bgr8"))
        self.rcv_times.append(rospy.get_rostime())
        self.rcv_frame_count += 1
        return self

    def find_windows(self):
        self.image_process = self.image_cv2.popleft()
        self.process_frame_count = self.rcv_frame_count - len(self.image_cv2)
        self.process_rcv_time = self.rcv_times.popleft()
        height, width, channels = self.image_process.shape

        # Detect objects
        blob = cv2.dnn.blobFromImage(self.image_process, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Show information
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                score = scores[class_id]
                if (score > self.tune.min_score) and (class_id == 0):
                    # Object detected
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    # Area
                    a = w * h
                    # Register data
                    boxes.append(self.box(x, y, w, h, a, cx, cy))
        self.filter_windows(boxes)
    
    def find_osnet_windows(self):
        
        self.osnet_windows = []
        for window in self.yolo_windows:
            # Consider desired aspect ratio 1:2 (w:h)
            diff = window.h - window.w*2
            h = 256
            w = 128
            if diff > 0:
                h = window.h
                w = window.h / 2
            elif diff < 0:
                h = window.w * 2
                w = window.w
            else:
                h = window.h
                w = window.w
            x = window.c_x - w/2
            y = window.c_y - h/2
            a = w * h

            # Move out-of-bounds windows
            x = min(x, self.image_process.shape[1] - w)
            x = max(0, x)
            y = min(y, self.image_process.shape[0] - h)
            y = max(0, y)
            h = min(self.image_process.shape[0], h)
            w = min(self.image_process.shape[1], w)

            box = self.box(x, y, w, h, a, window.c_x, window.c_y)
            self.osnet_windows.append(box)
                


    def plot_windows(self, level : int = 0):
        self.image_yolo = self.image_process

        if level <= 0:
            return
        height, width, channels = self.image_process.shape
        height = int(height/30)
        width = int(width/10)


        color = [255, 255, 255]

        for index in range(len(self.yolo_windows)):
            box = self.yolo_windows[index]
            txt = str(box.w) + 'x' + str(box.h)

            if (level == 3):
                color = colors[self.assignments[index]]
                txt = "Person " + str(self.assignments[index])
            cv2.rectangle(self.image_yolo, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
            cv2.rectangle(self.image_yolo, (box.x, box.y), (box.x + width, box.y + height), color, -1)
            cv2.putText(self.image_yolo, txt, (box.x, box.y + int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 0, 0], 1)

        if level == 1:
            return
        
        height, width, channels = self.image_process.shape
        self.image_pre_osnet = self.image_yolo
        for box in self.osnet_windows:
            x_start = int(max(0, box.x))
            x_end = int(min(width, box.x + box.w))
            y_start = int(max(0, box.y))
            y_end = int(min(height, box.y + box.h))
            cv2.rectangle(self.image_pre_osnet, (x_start, y_start), (x_end, y_end), [128, 128, 128], 1, cv2.LINE_4)
        
        if level == 2:
            return
        


    def filter_windows(self, boxes):
        # Picked indexes
        pick = []

        # Array of box coordintates
        x_start = []
        y_start = []
        x_end = []
        y_end = []
        a = []

        for box in boxes:
            x_start.append(box.x)
            y_start.append(box.y)
            x_end.append(box.x + box.w)
            y_end.append(box.y + box.h)
            a.append(box.a)
        
        x_start = np.array(x_start)
        y_start = np.array(y_start)
        x_end = np.array(x_end)
        y_end = np.array(y_end)
        a = np.array(a)

        # Sort boxes based on closeness to the camera
        indexes = np.argsort(y_end)
        # indexes = indexes.tolist()
        while len(indexes):
            end = len(indexes) - 1
            curr_index = indexes[end]
            # Pick the box
            pick.append(curr_index)
            
            # Find the largest overlapping box
            overlap_x_start = np.maximum(x_start[curr_index], x_start[indexes[:end]])
            overlap_y_start = np.maximum(y_start[curr_index], y_start[indexes[:end]])
            overlap_x_end = np.minimum(x_end[curr_index], x_end[indexes[:end]])
            overlap_y_end = np.minimum(y_end[curr_index], y_end[indexes[:end]])

            # Compute width and height of the overlapping box
            w = np.maximum(0, overlap_x_end - overlap_x_start + 1)
            h = np.maximum(0, overlap_y_end - overlap_y_start + 1)

            # Compute ratio of overlap
            overlap = (w * h) / a[indexes[:end]]

            # Delete indexes that go past the overlap threshold
            indexes = np.delete(indexes, np.concatenate(([end], np.where(overlap > self.tune.max_overlap)[0])))
        
        # Pick the windows
        self.yolo_windows = []
        for i in pick:
            self.yolo_windows.append(boxes[i])

    def extract_osnet_features(self):
        # Not accounting for windows out of frame bounds (yet)!

        window_scores = []

        self.assignments = np.full(len(self.osnet_windows),-1)
        # for gallery in self.features_gallery:
        #     gallery.assignment = -1
        
        descriptor = []
        try:
            for window in self.osnet_windows:
                cropped = self.image_process[int(window.x):int(window.x + window.w - 1), int(window.y):int(window.y + window.h - 1)]
                cropped = imutils.resize(cropped, width = 128, height = 256)
                descriptor.append(Frame.extractor(cropped))
                gallery_score = []
                for gallery in self.features_gallery:
                    scores = []
                    for element in gallery:
                        scores.append(distance.cosine(descriptor[len(descriptor)-1], element))
                    gallery_score.append(np.min(scores))
                window_scores.append(gallery_score)

            window_scores = np.matrix(window_scores)
            
            row_index, col_index = linear_sum_assignment(window_scores)

            for i in range(len(row_index)):
                if window_scores[row_index[i], col_index[i]] > 0.5:
                    continue
                self.assignments[row_index[i]] = col_index[i]
                if len(self.features_gallery[col_index[i]]) > 5:
                    self.features_gallery[col_index[i]].popleft()
                self.features_gallery[col_index[i]].append(descriptor[row_index[i]])
            
            for index in range(len(self.assignments)):
                if (self.assignments[index] == -1):
                    gal = deque()
                    gal.append(descriptor[index])
                    self.features_gallery.append(gal)
                    self.assignments[index] = len(self.features_gallery) - 1
            print(self.assignments)
                # print(type(descriptor))
                # print(descriptor.shape)

            
        except Exception as e:
            print(str(e))
        return




# Main function
def target_tracking():

    rate = rospy.Rate(10)
    T = rospy.Duration(0.1)
    
    model_path = os.path.realpath(os.path.join(osnet_path,'osnet_ain_x0_25_imagenet.pyth'))

    print("YOLO Path: ", yolo_path)
    print("Osnet Path: ", osnet_path)
    # Initialize frame instance
    try:
        frame = Frame(float(sys.argv[0]), float(sys.argv[1]))
    except ValueError:
        frame = Frame()
        
    #initialize time
    now = rospy.get_rostime()

    while not rospy.is_shutdown():
        # Wait for input before starting
        time.sleep(5)

        try:
            input("Press Enter to start target tracking")
        except SyntaxError:
            pass

        while True:
            if len(frame.image_cv2):
                frame.find_windows()
                frame.find_osnet_windows()
                frame.extract_osnet_features()
                frame.plot_windows(3)
                frame.publisher.publish(bridge.cv2_to_imgmsg(frame.image_pre_osnet))
                msg = "Processed frame #" + str(frame.process_frame_count) + " in " + str((rospy.get_rostime() - frame.process_rcv_time).to_sec()) + "s "
                print(msg)
                frame.logs.publish(msg)





#   Guard
if __name__ == '__main__':    
    
    # Initialize node
    rospy.init_node("target_tracking", anonymous=True)
    rospy.loginfo("ROS Node up!")

    # Launch in execution
    try:
        target_tracking()
    except rospy.ROSInterruptException:
        pass
