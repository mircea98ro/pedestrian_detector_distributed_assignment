#!/usr/bin/env python3

#   Imports

import os
from dataclasses import dataclass

import cv2
import numpy
import rospy
from cv2 import dnn_Net
from cv_bridge import CvBridge
from detection_osnet.msg import ProcessWindow, Window, WindowPack
from sensor_msgs.msg import CompressedImage


#   Globals

bridge = CvBridge()


#   Classes

class YOLO:
    #   Nested classes
    class YOLOFiles:
        def __init__(self, path: str, weight: str, cfg: str, names: str):
            self.path = path
            self.weight = weight
            self.cfg = cfg
            self.names = names

        def getPath(self, filetype: str):
            if filetype == "weight":
                aux = self.weight
            elif filetype == "cfg":
                aux = self.cfg
            elif filetype == "names":
                aux = self.names
            return os.path.realpath(os.path.join(self.path, aux))

    @dataclass
    class YOLONetwork:
        net: dnn_Net
        classes: list
        output_layers: list

    @dataclass
    class YOLOParameters:
        min_score: float
        max_overlap: float
        min_box_confidence: float
        max_count: int

    def __init__(self, params: YOLOParameters, files: YOLOFiles):
        self.dataS = rospy.Subscriber('camera/color/image_raw/compressed', CompressedImage, self.callback, queue_size=1)
        self.dataP = rospy.Publisher('processing/yolo', WindowPack, queue_size=1)
        self.pending = False

        self.rcv : CompressedImage = None

        rospy.loginfo("YOLO network initializing...")
        self.params = params
        self.files = files
        self.network = self.YOLONetwork(None, None, None)
        self.network.net = cv2.dnn.readNet(
            self.files.getPath('weight'), self.files.getPath('cfg'))
        self.network.classes = []
        with open(self.files.getPath('names'), "r") as f:
            self.network.classes = [line.strip() for line in f.readlines()]
        layer_names = self.network.net.getLayerNames()
        self.network.output_layers = [layer_names[i - 1]
                                      for i in self.network.net.getUnconnectedOutLayers()]
        rospy.loginfo("YOLO network initialized succesfully!")

    def detect(self):
            img = bridge.compressed_imgmsg_to_cv2(self.rcv, "bgr8")
            height, width, channels = img.shape
            # YOLO Detection
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.network.net.setInput(blob)
            outs = self.network.net.forward(self.network.output_layers)

            boxes = []
            for out in outs:
                for detection in out:
                    if (detection[4] < self.params.min_box_confidence):
                        continue
                    scores = detection[5:]
                    class_id = numpy.argmax(scores)
                    score = scores[class_id]
                    if (score > self.params.min_score) and (class_id == 0):
                        # Object detected
                        x = int(detection[0] * width)
                        y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Register data
                        boxes.append(Window(x = x, y = y, w = w, h = h))
            # Filter data
            windows = []
            for i in self.filter_windows(boxes):
                windows.append(ProcessWindow(window = boxes[i], assignment = None))
            msg = WindowPack(data = windows, img = self.rcv)
            msg.header.stamp = self.rcv.header.stamp
            msg.timestamp = rospy.Time.now()
            self.dataP.publish(msg)
            self.pending = False
            
            
    def filter_windows(self, boxes: list):
        # Picked indexes
        pick = []

        # Array of box coordintates
        x_start = []
        y_start = []
        x_end = []
        y_end = []
        a = []

        for box in boxes:
            dh = int(box.h/2)
            dw = int(box.w/2)
            x_start.append(box.x - dw)
            y_start.append(box.y - dh)
            x_end.append(box.x + dw)
            y_end.append(box.y + dh)
            a.append(box.h*box.w)

        x_start = numpy.array(x_start)
        y_start = numpy.array(y_start)
        x_end = numpy.array(x_end)
        y_end = numpy.array(y_end)
        a = numpy.array(a)

        # Sort boxes based on closeness to the camera
        indexes = numpy.argsort(y_end)
        # indexes = indexes.tolist()
        while len(indexes):
            end = len(indexes) - 1
            curr_index = indexes[end]
            # Pick the box
            pick.append(curr_index)

            # Find the largest overlapping box
            overlap_x_start = numpy.maximum(
                x_start[curr_index], x_start[indexes[:end]])
            overlap_y_start = numpy.maximum(
                y_start[curr_index], y_start[indexes[:end]])
            overlap_x_end = numpy.minimum(
                x_end[curr_index], x_end[indexes[:end]])
            overlap_y_end = numpy.minimum(
                y_end[curr_index], y_end[indexes[:end]])

            # Compute width and height of the overlapping box
            w = numpy.maximum(0, overlap_x_end - overlap_x_start + 1)
            h = numpy.maximum(0, overlap_y_end - overlap_y_start + 1)

            # Compute ratio of overlap
            overlap = (w * h) / a[indexes[:end]]

            # Delete indexes that go past the overlap threshold
            indexes = numpy.delete(indexes, numpy.concatenate(
                ([end], numpy.where(overlap > self.params.max_overlap)[0])))

        # Pick the windows
        return pick


    def callback(self, msg : CompressedImage):
        if self.pending == True:
            return
        # if msg.header.stamp.is_zero():
        msg.header.stamp = rospy.Time.now()
        self.rcv = msg
        self.pending = True
        

#   Functions

def yolo():

    files = YOLO.YOLOFiles(rospy.get_param("cfg/yolo/path"), rospy.get_param("cfg/yolo/weight"), rospy.get_param("cfg/yolo/cfg"), rospy.get_param("cfg/yolo/names"))
    params = YOLO.YOLOParameters(rospy.get_param("cfg/yolo/min_score"), rospy.get_param("cfg/yolo/max_overlap"), rospy.get_param("cfg/yolo/min_box_confidence"), rospy.get_param("cfg/yolo/max_count"))
    
    obj = YOLO(params = params, files = files)
    rospy.on_shutdown(world_end)

    rospy.loginfo("Robot YOLO node running!")

    while not rospy.is_shutdown():
        if (obj.pending == True):
            obj.detect()

def world_end():
    rospy.loginfo("Robot YOLO node shutting down.")


#   Guard

if __name__ == '__main__':    
    
    # Initialize node
    rospy.init_node("yolo", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Robot YOLO Node initializing...")

    # Launch in execution
    try:
        yolo()
    except rospy.ROSInterruptException:
        rospy.logerr("Robot YOLO Node initialization error!")
        pass
