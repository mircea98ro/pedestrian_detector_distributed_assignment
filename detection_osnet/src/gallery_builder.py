#!/usr/bin/env python3

#   Imports

import datetime
import os
from tkinter import filedialog

import cv2
import numpy
os.environ['ROS_NAMESPACE'] = 'r_1'   # Uncomment to force node namespace
import rospy
import torch
from detection_osnet.msg import Window

from reid import ReID
from yolo import YOLO

#   Defines


#   Globals

#   Classes

#   Functions

def gallery_builder():
    try:
        target_no = rospy.get_param("/master/target_no")

        files = YOLO.YOLOFiles(
            rospy.get_param("cfg/yolo/path"), 
            rospy.get_param("cfg/yolo/weight"), 
            rospy.get_param("cfg/yolo/cfg"), 
            rospy.get_param("cfg/yolo/names")
            )
        params = YOLO.YOLOParameters(
            rospy.get_param("cfg/yolo/min_score"), 
            rospy.get_param("cfg/yolo/max_overlap"), 
            rospy.get_param("cfg/yolo/min_box_confidence"), 
            rospy.get_param("cfg/yolo/max_count")
            )
        yolo_obj = YOLO(params = params, files = files)

        files = ReID.ReIDFiles(
            rospy.get_param("cfg/reid/path"), 
            rospy.get_param("cfg/reid/model")
            )
        params = ReID.ReIDParameters(
            rospy.get_param("cfg/reid/descriptor_size"), 
            rospy.get_param("cfg/reid/window_w"), 
            rospy.get_param("cfg/reid/window_ratio"), 
            rospy.get_param("cfg/reid/max_distance"), 
            rospy.get_param("cfg/reid/dynamic_gallery"), 
            rospy.get_param("/master/target_no"), 
            rospy.get_param("cfg/reid/redo_yolo_windows"),
            0
            )
        reid_obj = ReID(files, params)


    except Exception:
        rospy.logerr("Configuration info missing! Load configuration file / Run configuration script, then try again!")
        rospy.signal_shutdown("Node cannot run without configuration info!")
        return
    
    # names = []
    # for id in range(target_no):
    #     name = input(f'Please enter nickname for gallery {id + 1}: ')
    #     names.append(name)
    filetypes = (('JPEG Image', '*.jpg'), ('PNG Image', '*.png'))
    rospy.loginfo("Load images to create gallery...")

    filenames = filedialog.askopenfilenames(
        title='Select images...',
        initialdir= os.path.expanduser('~'),
        filetypes=filetypes)
    
    dir = os.path.dirname(filenames[0])
    exportPath = f'{dir}/../Galleries'
    
    try:
        os.makedirs(exportPath)
    except Exception as e:
        rospy.logwarn(e)
    
    try:
        os.chdir(exportPath)
    except Exception as e:
        rospy.logerr(e)
        return
    
    # for name in names:
    #     try:
    #         os.makedirs(f'{name}')
    #     except Exception as e:
    #         rospy.logwarn(e)
    

    try:
        os.makedirs(f'unorganised')
    except Exception as e:
        rospy.logwarn(e)

    counter = 0
    for fn in filenames:
        counter += 1
        img = cv2.imread(fn)
        height, width, channels = img.shape

        # YOLO Detection
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_obj.network.net.setInput(blob)
        outs = yolo_obj.network.net.forward(yolo_obj.network.output_layers)

        boxes = []
        for out in outs:
            for detection in out:
                if (detection[4] < yolo_obj.params.min_box_confidence):
                    continue
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                score = scores[class_id]
                if (score > yolo_obj.params.min_score) and (class_id == 0):
                    # Object detected
                    x = int(detection[0] * width)
                    y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Register data
                    boxes.append(Window(x = x, y = y, w = w, h = h))
        # Filter data
        windows = []
        for i in yolo_obj.filter_windows(boxes):
            windows.append(boxes[i])

        # if len(windows) != target_no:
        #     rospy.logwarn(f'Image ({fn}) did not return enough recognized persons and was skipped!')
        #     continue
        windows.sort(key=sort_key)

        for i in range(len(windows)):
            xLeft = int(max(0, windows[i].x - windows[i].w/2))
            yUp = int(max(0, windows[i].y - windows[i].h/2))
            xRight = int(min(width, windows[i].x + windows[i].w/2 - 1))
            yDown = int(min(height, windows[i].y + windows[i].h/2 - 1))

            base = os.path.splitext(os.path.basename(fn))[0]

            cropped = img[yUp:yDown, xLeft:xRight]
            cv2.imwrite(f'unorganised/{base}_{i}.jpg',cropped)
            
            desc = reid_obj.extractor(cropped)
            # torch.save(desc, f'{names[i]}/{base}.pt')
            torch.save(desc, f'unorganised/{base}_{i}.pt')
    pass

def sort_key(w : Window):
    return w.x
#   Guard

if __name__ == '__main__':    
    
    # Initialize node
    rospy.init_node("gallery_builder", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Gallery Builder Node initializing...")

    # Launch in execution
    try:
        gallery_builder()
    except rospy.ROSInterruptException:
        rospy.logerr("Gallery Builder Node initialization error!")
        pass
