#!/usr/bin/env python3

# Run this script (not using roslaunch!) to set parameters. Then use `rosparam dump filename.yaml /cfg` to create a param file for loading later

import os
import rospy

def cfg_creator():
    rospy.set_param('cfg/yolo/path', os.path.join(os.path.dirname(__file__), '..', '..', 'detection', 'darknet'))
    rospy.set_param('cfg/yolo/weight', 'yolov3-tiny.weights')
    rospy.set_param('cfg/yolo/cfg', 'yolov3-tiny.cfg')
    rospy.set_param('cfg/yolo/names', 'coco.names')

    rospy.set_param('cfg/yolo/max_iou', 0.8)
    rospy.set_param('cfg/yolo/min_obj_confidence', 0.2)
    rospy.set_param('cfg/yolo/min_score', 0.1)

    rospy.set_param('cfg/reid/path', os.path.join(os.path.dirname(__file__), '..', 'osnet_data'))
    rospy.set_param('cfg/reid/model', 'osnet_ain_x0_25')

    rospy.set_param('cfg/reid/descriptor_size', 1)
    rospy.set_param('cfg/reid/window_w', 128)
    rospy.set_param('cfg/reid/window_ratio', 2)
    rospy.set_param('cfg/reid/max_distance', 0.6)
    rospy.set_param('cfg/reid/redo_yolo_windows', 0)
    rospy.set_param('cfg/reid/dynamic_gallery', 0)
    rospy.set_param('cfg/reid/load_galleries', 0)

    rospy.set_param('cfg/kalman/source', 'osnet')

    rospy.set_param('/master/robot_no', 1)
    rospy.set_param('/master/target_no', 6)
    rospy.set_param('/master/export', 0)

#   Guard
if __name__ == '__main__':

    # Launch in execution
    try:
        cfg_creator()
    except Exception:
        rospy.logerr("Config generation error!")
        pass