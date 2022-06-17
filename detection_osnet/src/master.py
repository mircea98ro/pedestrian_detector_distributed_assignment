#!/usr/bin/env python3

# Monitor (master) node. Can read images and data from robot. Compiles statistics and presents data for visualisation
#
# TFM - Mircea-Alexandru Popovici
# Directors - Eduardo Montijano, Danilo Tardioli, Adrian Burlacu



#   Imports

import csv                      # Read/Write CSV Files
import datetime                 # Get Date and DateTime system info
import os                       # OS Tools
from typing import List         # Typed lists

import cv2                      # OpenCV
import rospy                    # ROS-Python Interface
from cv_bridge import CvBridge  # Interface between OpenCV and ROS imaging
from scipy.io import savemat    # Read/Write MAT Files

from monitor import Monitor     # Monitor class

#   Defines

#   Globals

bridge = CvBridge()

class Master:

    def __init__(self):
        self.robot_no = rospy.get_param('/master/robot_no')
        self.does_export = rospy.get_param('master/export')

        if self.does_export:
            home = os.path.expanduser('~')
            self.exportPath = f'{home}/PDDA_Results/Run_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
            
            try:
                os.makedirs(self.exportPath)
            except Exception as e:
                rospy.logwarn(e)
            
            try:
                os.chdir(self.exportPath)
            except Exception as e:
                rospy.logerr(e)
                return
            
            for id in range(self.robot_no):
                try:
                    os.makedirs(f'yolo/img')
                    os.makedirs(f'osnet/img')
                    os.makedirs(f'kalman/img')
                except Exception as e:
                    rospy.logwarn(e)
        
        self.monitors : List["Monitor"] = []

        colors = [  [68, 65, 249],
                    [142, 144, 77],
                    [79, 199, 249],
                    [44, 114, 243],
                    [144, 117, 87],
                    [109, 190, 144],
                    [30, 150, 248],
                    [161, 125, 39],
                    [139, 170, 67],
                    [255, 255, 255]
                ]

        try:
            self.monitors.append(Monitor(bridge = bridge, colors = colors, ns = 1, target_no=rospy.get_param('/master/target_no'), source = 'yolo'))
            self.monitors.append(Monitor(bridge = bridge, colors = colors, ns = 1, target_no=rospy.get_param('/master/target_no'), source = 'osnet'))
            self.monitors.append(Monitor(bridge = bridge, colors = colors, ns = 1, target_no=rospy.get_param('/master/target_no'), source = 'kalman'))
        except Exception:
            rospy.logerr("Configuration info missing! Load configuration file / Run configuration script, then try again!")
            rospy.signal_shutdown("Node cannot run without configuration info!")
            return
            
    def stats(self):
        id = 0
        for m in self.monitors:
            if m.active == False:
                continue
            pt_mean, pt_std, correct, wrong = m.summary()
            txt = f'{m.source} summary: \n\tAverage processing time: {pt_mean.to_sec()}s \u00B1 {pt_std.to_sec()}s\n\tAccuracy: {correct}%\n'
            rospy.loginfo(txt)
            if (self.does_export):
                with open(f'{m.source}/summary.csv', 'w') as f:
                    writer = csv.writer(f, delimiter = '\t')
                    writer.writerow(['Description', 'avg.', 'std.'])
                    writer.writerow(['Processing time', f'{pt_mean.to_sec()}s', f'{pt_std.to_sec()}s'])
                    writer.writerow(['Accuracy', correct/100, 0])
                with open(f'{m.source}/summary.txt', 'w') as f:
                    f.writelines(
                                [f'Avg. processing time = {pt_mean.to_sec()}s \u00B1 {pt_std.to_sec()}s\n',
                                f'Accuracy = {correct}%']
                                )
                mdic = {'mean_process_time' : pt_mean.to_sec(), 'std_process_time' : pt_std.to_sec(), 'accuracy' : correct / 100}
                savemat(f'{m.source}/summary.mat', mdic, appendmat=False)

    def service(self):
        for m in self.monitors:
            if len(m.rcv) > 0:
                img = m.service()
                if (self.does_export):
                    cv2.imwrite(f'{m.source}/img/frame_{m.process_frame_count:06}.jpg', img)
                    

#   Functions

def master():

    # Initialization
    master = Master()

    # Return summary on shutdown
    rospy.on_shutdown(master.stats)

    # Mark node as active after init
    if (not rospy.is_shutdown()):
        rospy.loginfo("Master node active. ")

    # Normal operation
    while not rospy.is_shutdown():
        master.service()


#   Guard
if __name__ == '__main__':    
    
    # Initialize node
    rospy.init_node("master", anonymous=False, log_level = rospy.DEBUG)
    rospy.loginfo("Master node initializing!")

    # Launch in execution
    try:
        master()
    except rospy.ROSInterruptException:
        rospy.logerr("Master node initialization error!")
        pass
