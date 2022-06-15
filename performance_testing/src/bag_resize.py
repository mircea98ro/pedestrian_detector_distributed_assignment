#!/usr/bin/env python3

#   Imports

import os
import sys
import tkinter
from tkinter import filedialog
import cv2
from cv_bridge import CvBridge
import rospy
import rosbag

#   Global

bridge = CvBridge()

#   Functions

def bag_resizer(src : str, dest : str):
    inBag = rosbag.Bag(src, 'r')
    outBag = rosbag.Bag(dest, 'w')
    
    w = int(sys.argv[1])
    h = int(sys.argv[2])
    dims = (w, h)

    rospy.loginfo("Bag Resizer initialized!")
    
    for topic, msg, t in inBag.read_messages():
        if 'image_raw' in topic:
            img = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            resized = cv2.resize(img, dims, interpolation=cv2.INTER_CUBIC)
            imgmsg = bridge.cv2_to_compressed_imgmsg(resized, dst_format='jpg')
            outBag.write(f'{topic}', imgmsg, t)
    rospy.loginfo("Resizing done! Closing.")
    pass


#   Guard

if __name__ == '__main__':
    rospy.init_node("converter", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Bag Resizer initializing...")
    filetypes = (('ROS Bag files', '*.bag'), ('All files', '*.*'))
    src = filedialog.askopenfilename(
                    title=f'Select input ROS Bag',
                    initialdir=os.path.realpath(__file__),
                    filetypes=filetypes
                    )
                
    dest = filedialog.asksaveasfilename(
                    title=f'Save ROS Bag as...',
                    initialdir=os.path.dirname(src),
                    filetypes=filetypes
    )
    
    # Launch in execution
    try:
        bag_resizer(src, dest)
    except rospy.ROSInterruptException:
        rospy.logerr("Bag Resizer Node initialization error!")
        pass
