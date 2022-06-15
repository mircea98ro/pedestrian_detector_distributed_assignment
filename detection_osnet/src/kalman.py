#!/usr/bin/env python3

#   Imports

from cmath import inf
import os
from typing import List
os.environ["ROS_NAMESPACE"] = "/r_1"
from dataclasses import dataclass

import cv2
from cv2 import KalmanFilter
import numpy
import rospy
from cv2 import dnn_Net
from detection_osnet.msg import ProcessWindow, Window, WindowPack
from sensor_msgs.msg import Image


#   Globals



#   Classes

class Kalman:

    @dataclass
    class KalmanWindow:
        kf : KalmanFilter
        window : Window
        missing : int

    def __init__(self, source : str):
        self.dataS = rospy.Subscriber(f'processing/{source.lower()}', WindowPack, self.callback, queue_size=1)
        self.dataP = rospy.Publisher('processing/kalman', WindowPack, queue_size=1)

        self.pending = False
        self.rcv : WindowPack = None

        self.next : List['self.KalmanWindow'] = []

    def process(self):

        for kw in self.next:
            kw.missing += 1
        window : ProcessWindow
        for  window in self.rcv.data:
            while window.assignment > len(self.next):
                self.next.append(self.KalmanWindow(kf = KalmanFilter(4,2), window = None, missing = inf))
                self.next[len(self.next) - 1].kf.measurementMatrix = numpy.array([[1, 0, 0, 0], 
                                                                           [0, 1, 0, 0]], numpy.float32)
                self.next[len(self.next) - 1].kf.transitionMatrix = numpy.array([[1, 0, 1, 0], 
                                                                          [0, 1, 0, 1], 
                                                                          [0, 0, 1, 0], 
                                                                          [0, 0, 0, 1]], numpy.float32)
                self.next[len(self.next) - 1].kf.processNoiseCov = numpy.eye(4, dtype = numpy.float32) * 0.03
            self.next[window.assignment - 1].window = window.window
            self.next[window.assignment - 1].missing = 0

            aux = numpy.array((self.next[window.assignment - 1].window.x, self.next[window.assignment - 1].window.y), numpy.float32)
            self.next[window.assignment - 1].kf.correct(aux)

        msg = WindowPack(img = self.rcv.img)

        for kw in self.next:
            if kw.missing < 20:
                measurement = kw.kf.predict()
                kw.window.x = measurement[0]
                kw.window.y = measurement[1]
                msg.data.append(ProcessWindow(window = kw.window, assignment = self.next.index(kw) + 1))

        msg.header.stamp = self.rcv.header.stamp
        msg.timestamp = rospy.Time.now()
        self.dataP.publish(msg)
        self.pending = False

    def callback(self, msg : WindowPack):
        if self.pending == True:
            return
        self.rcv = msg
        self.pending = True
        

#   Functions

def kalman():

       
    obj = Kalman(source = rospy.get_param("cfg/kalman/source"))

    rospy.on_shutdown(world_end)

    rospy.loginfo("Robot Kalman node running!")

    while not rospy.is_shutdown():
        if obj.pending:
            obj.process()

def world_end():
    rospy.loginfo("Robot Kalman node shutting down.")


#   Guard

if __name__ == '__main__':    
    
    # Initialize node
    rospy.init_node("kalman", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Robot Kalman Node initializing...")

    # Launch in execution
    try:
        kalman()
    except rospy.ROSInterruptException:
        rospy.logerr("Robot Kalman Node initialization error!")
        pass
