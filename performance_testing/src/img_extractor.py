#!/usr/bin/env python3

#   Imports

from tkinter import filedialog
import cv2
from cv_bridge import CvBridge
import rospy
import rosbag
import os
import genpy

#   Global

bridge = CvBridge()

#   Functions

def img_extractor():
    
    filetypes = (('ROS Bag files', '*.bag'), ('All files', '*.*'))
    src = filedialog.askopenfilename(
                    title=f'Select input ROS Bag',
                    initialdir=os.path.realpath(__file__),
                    filetypes=filetypes
                    )

    inBag = rosbag.Bag(src, 'r')

    home = os.path.expanduser('~')
    base = os.path.basename(src)
    exportPath = f'{home}/PDDA_Extracts/{os.path.splitext(base)[0]}/Images'
    
    try:
        os.makedirs(exportPath)
    except Exception as e:
        rospy.logwarn(e)
    
    try:
        os.chdir(exportPath)
    except Exception as e:
        rospy.logerr(e)
        return

    rospy.loginfo("Image extractor initialized!")
    
    i = 0
    topic : str
    t : genpy.Time
    for topic, msg, t in inBag.read_messages():
        if 'compressed' in topic:
            img = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        else:
            img = bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imwrite(f'img_{i}.jpg', img)
        i += 1

    
    rospy.loginfo("Extraction done! Closing.")
    pass


#   Guard

if __name__ == '__main__':
    rospy.init_node("extractor", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Image extractor initializing...")
    # Launch in execution
    try:
        img_extractor()
    except rospy.ROSInterruptException:
        rospy.logerr("Image extractor initialization error!")
        pass
