#!/usr/bin/env python3

#   Imports

import cv2
from cv_bridge import CvBridge
import rospy
import rosbag

#   Global

bridge = CvBridge()

#   Functions

def bag_convert(src : str, dest : str):
    inBag = rosbag.Bag(src, 'r')
    outBag = rosbag.Bag(dest, 'w')

    rospy.loginfo("Bag Converter initialized!")
    for topic, msg, t in inBag.read_messages():
        if 'image_raw' in topic:
            imgmsg = bridge.cv2_to_compressed_imgmsg(bridge.imgmsg_to_cv2(msg, 'bgr8'), dst_format='jpg')
            outBag.write(f'{topic}/compressed', imgmsg, t)
    
    rospy.loginfo("Conversion done! Closing.")
    pass


#   Guard

if __name__ == '__main__':
    rospy.init_node("converter", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Bag Converter initializing...")
    src = rospy.get_param('converter/src')
    dest = rospy.get_param('converter/dest')
    # Launch in execution
    try:
        bag_convert(src, dest)
    except rospy.ROSInterruptException:
        rospy.logerr("Bag Converter Node initialization error!")
        pass
