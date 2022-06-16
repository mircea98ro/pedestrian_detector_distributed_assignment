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
        max_iou: float
        min_obj_confidence: float

    @dataclass
    class ScoredWindow:
        window : Window
        score : float
        area : float
    
    def score_sortkey(sw : ScoredWindow):
        return sw.score

    def y_sortkey(sw : Window):
        return sw.y + sw.h/2
    def x_sortkey(sw : ScoredWindow):
        return sw.x - sw.w/2
    
    def __init__(self, params: YOLOParameters, files: YOLOFiles, target_no : int):
        self.dataS = rospy.Subscriber('camera/color/image_raw/compressed', CompressedImage, self.callback, queue_size=1)
        self.dataP = rospy.Publisher('processing/yolo', WindowPack, queue_size=1)
        self.pending = False

        self.rcv : CompressedImage = None
        self.target_no = target_no
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
                    if (detection[4] < self.params.min_obj_confidence):
                        continue
                    scores = detection[5:]
                    class_id = numpy.argmax(scores)
                    if (class_id != 0):
                        continue

                    score = scores[class_id] * detection[4]
                    if (score  > self.params.min_score):
                        # Object detected
                        x = int(detection[0] * width)
                        y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Register data
                        boxes.append(self.ScoredWindow(window = Window(x = x, y = y, w = w, h = h), score = score, area = w*h))
            # Filter data
            windows = []
            for w in self.filter_windows(boxes):
                windows.append(ProcessWindow(window = w, assignment = None))
            msg = WindowPack(data = windows, img = self.rcv)
            msg.header.stamp = self.rcv.header.stamp
            msg.timestamp = rospy.Time.now()
            self.dataP.publish(msg)
            self.pending = False
            
            
    def filter_windows(self, boxes: list):

        boxes.sort(key = self.score_score_sortkey)
        
        # Picked indexes
        pick = []

        for i in range(0, len(boxes) - 1):
            should_select = True
            for j in range(i+1, len(boxes)):
                if (self.iou(boxes[i], boxes[j]) > self.params.max_iou):
                    should_select = False
                    break
            if should_select:
                pick.append(boxes[i].window)
        pick.sort(key=self.x_sortkey,reverse=False)
        
        # Pick the windows
        return pick

    def iou(self, sw1 : ScoredWindow, sw2 : ScoredWindow):
        mx = min(sw1.window.x-sw1.window.w/2, sw2.window.x-sw2.window.w/2)
        Mx = max(sw1.window.x+sw1.window.w/2, sw2.window.x+sw2.window.w/2)


        my = min(sw1.window.y-sw1.window.h/2, sw2.window.y-sw2.window.h/2)
        My = max(sw1.window.y+sw1.window.h/2, sw2.window.y+sw2.window.h/2)

        i = (Mx - mx) * (My - my)
        u = sw1.area + sw2.area - i
        return i/u

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
    params = YOLO.YOLOParameters(rospy.get_param("cfg/yolo/min_score"), rospy.get_param("cfg/yolo/max_iou"), rospy.get_param("cfg/yolo/min_obj_confidence"), rospy.get_param("cfg/yolo/max_count"))
    target_no = rospy.get_param("/master/target_no")
    obj = YOLO(params = params, files = files, target_no = target_no)
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
