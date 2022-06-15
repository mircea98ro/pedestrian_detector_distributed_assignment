#!/usr/bin/env python3

#   Imports

from math import sqrt                                           # Square root function
from typing import Deque, List, NamedTuple                      # Typed data structures

import cv2                                                      # OpenCV
import numpy                                                    # Numpy for math functions
import rospy                                                    # ROS - Python interface
from cv_bridge import CvBridge                                  # Interface between OpenCV and ROS imaging
from detection_osnet.msg import (Accuracy, MonitorUpdate, 
                                 ProcessWindow, WindowPack)     # Custom message types used by node I/O
from sensor_msgs.msg import CompressedImage                     # ROS compressed image message type

#   Defines

DETECT_LVL_RAW = 0
DETECT_LVL_YOLO = 1
DETECT_LVL_OSNET = 2
DETECT_LVL_KALMAN = 3

#   Classes

class Monitor:


    class Results(NamedTuple):
        avg_process_time : rospy.Duration
        std_process_time : rospy.Duration
        avg_correct_percent : numpy.float64
        avg_wrong_percent : numpy.float64

    class Writer:

        def __init__(self, ns: str = None, post_queue_len: int = 1):
            self.imgP = rospy.Publisher(f'/r_{ns}/results/img/compressed', CompressedImage, queue_size=post_queue_len)
            self.dataP = rospy.Publisher(f'r_{ns}/results', MonitorUpdate, queue_size=post_queue_len)
        
        def write_frame(self, img : CompressedImage, data : MonitorUpdate):
            self.imgP.publish(img)
            self.dataP.publish(data)
            # rospy.loginfo("Monitor TX!")

    def __init__(self, bridge : CvBridge, colors : numpy.array = None, ns : str = None, data_queue_len: int = 1, raw_img_queue_len: int = 1, post_queue_len: int = 1, target_no : int = 0, source : str = None):
        

        self.rcv : Deque['WindowPack'] = Deque()
        self.reader = rospy.Subscriber(f'/r_{ns}/processing/{source}', WindowPack, self.callback, queue_size = 1)

        self.writer = self.Writer(ns, post_queue_len)

        self.bridge = bridge
        self.target_no = target_no
        self.colors = colors

        if source == 'yolo':
            self.level = DETECT_LVL_YOLO
        elif source == 'osnet':
            self.level = DETECT_LVL_OSNET
        elif source == 'kalman':
            self.level = DETECT_LVL_KALMAN

        self.process_time : List['rospy.Duration'] = []
        self.total_process_time = rospy.Duration()
        self.accuracy = Accuracy()

        self.process_frame_count = 0

        self.first_frame_id = 0
        self.last_frame_id = 0

        

    def callback(self, msg : WindowPack):
        self.rcv.append(msg)

    def service(self):

        now = rospy.Time.now()
        msg = self.rcv.popleft()

        # Processing time
        aux = msg.timestamp - msg.header.stamp
        self.process_time.append(aux)
        self.total_process_time += aux

        # Frame Loss
        self.process_frame_count += 1
        
        if self.first_frame_id == 0:
            self.first_frame_id = msg.img.header.seq
            lost_frames = 0
        else:
            lost_frames = msg.img.header.seq - self.last_frame_id
        self.last_frame_id = msg.img.header.seq

        # Accuracy
        if (len(msg.data) != self.target_no):
            self.accuracy.wrong += 1
        else:
            self.accuracy.correct += 1
            if self.level > DETECT_LVL_YOLO: # For tests where each person keeps its lane
                pw = sorted(msg.data, key = sortKeyProcessWindow) # Sort windows from leftmost to the right
                for i in range(len(pw)):
                    if (pw[i].assignment != i+1):
                        self.accuracy.wrong += 1
                        self.accuracy.correct -= 1
                        break


        # Process Image
        img_cv2 = self.bridge.compressed_imgmsg_to_cv2(msg.img,"bgr8")

        if self.level > DETECT_LVL_RAW:
            height, width, channels = img_cv2.shape
            color = [255, 255, 255] # White, default
            txtbox_h = int(height/30)

            pw : ProcessWindow
            for pw in msg.data:
                xLeft = int(max(0, pw.window.x - pw.window.w/2))
                yUp = int(max(0, pw.window.y - pw.window.h/2))
                xRight = int(min(width, pw.window.x + pw.window.w/2 - 1))
                yDown = int(min(height, pw.window.y + pw.window.h/2 - 1))
                
                txtbox_w = pw.window.w

                txt = str(pw.window.w) + 'x' + str(pw.window.h)

                if self.level > DETECT_LVL_YOLO:
                    if (pw.assignment <= self.target_no):
                        color = self.colors[pw.assignment - 1]
                        txt = f'Person {pw.assignment}'
                    else:
                        color = [255, 255, 255] # White, default
                        txt = f'Person ?'

                
                # Main YOLO Window
                cv2.rectangle(img_cv2, (xLeft, yUp), (xRight, yDown), color, 2)

                # YOLO Textbox
                cv2.rectangle(img_cv2, (xLeft, yDown - txtbox_h + 1), (xRight, yDown), color, -1)

                # YOLO Text
                txtcolor = 0
                cv2.putText(img_cv2, txt, (xLeft + 1, yDown - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txtcolor, 1)

                img = self.bridge.cv2_to_compressed_imgmsg(img_cv2, "jpg")

                self.writer.write_frame(img = img, data = MonitorUpdate(accuracy = self.accuracy, frame_loss = lost_frames, avg_processing_time = self.total_process_time/self.process_frame_count, processing_time = self.process_time[self.process_frame_count - 1]))

        return img_cv2

    def summary(self):
        try:
            mean_processing_time : rospy.Duration = rospy.Duration.from_sec(0)
            std_processing_time : rospy.Duration = rospy.Duration.from_sec(0)
            aux_std : numpy.float = 0
            
            for t in self.process_time:
                mean_processing_time += t
            mean_processing_time /= self.process_frame_count
            
            mean_tosec = mean_processing_time.to_sec()
            for t in self.process_time:
                aux_std += (t.to_sec() - mean_tosec)**2
            aux_std /= self.process_frame_count - 1
            aux_std = sqrt(aux_std)
            std_processing_time = rospy.Duration.from_sec(aux_std)
        except ValueError:
            pass
        
        try:
            avg_correct = numpy.true_divide(self.accuracy.correct, self.process_frame_count)*100
            avg_wrong = numpy.true_divide(self.accuracy.wrong, self.process_frame_count)*100
        except Exception:
            avg_correct = 0
            avg_wrong = 0

        return self.Results(avg_process_time = mean_processing_time, 
                            std_process_time = std_processing_time,
                            avg_correct_percent = avg_correct, 
                            avg_wrong_percent = avg_wrong
                            )


#   Functions

def sortKeyProcessWindow(pw : ProcessWindow):
    return pw.window.x
