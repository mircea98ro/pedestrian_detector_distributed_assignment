#!/usr/bin/env python3

#   Imports

import os                                       # OS Tools
from collections import deque                   # Double-ended queue data structure
from dataclasses import dataclass               # C-type structs
from tkinter import filedialog                  # 'Open file(s)' dialog UI
from typing import List                         # Typed lists

import imutils                                  # Image processing utility package
import numpy                                    # Math library
# os.environ['ROS_NAMESPACE'] = 'r_1'           # Uncomment to force node namespace
import rospy                                    # ROS-Python Interface
import torch                                    # PyTorch for working with neural networks & tensors
from cv_bridge import CvBridge                  # Interface between OpenCV and ROS imaging
from detection_osnet.msg import WindowPack      # Output message data type
from scipy.optimize import \
    linear_sum_assignment                       # Hungarian algorithm (bipartite assignment)
from scipy.spatial import distance              # Cosine distance
from torchreid.utils import FeatureExtractor    # OSnet feature extractor for re-identification

#   Globals

bridge = CvBridge()


#   Classes

class ReID:
    #   Nested classes
    class ReIDFiles:
        def __init__(self, path: str, model: str):
            self.path = path
            self.model = model

        def getPath(self):
            return os.path.realpath(os.path.join(self.path, self.model + "_imagenet.pyth"))

    @dataclass
    class ReIDParameters:
        descriptor_size: int
        window_w: int
        window_ratio: float
        max_distance: float
        dynamic_gallery_content : bool
        targets : int
        redo_windows : bool
        load_galeries : bool

    @dataclass
    class ReIDGallery:
        descriptors: list

    #   Functions

    def __init__(self, files: ReIDFiles, params: ReIDParameters = None, galleries: list = None, device: str = 'cpu'):

        self.dataS = rospy.Subscriber('processing/yolo', WindowPack, self.callback, queue_size = 1)

        self.dataP = rospy.Publisher('processing/osnet', WindowPack, queue_size=1)

        self.wp = None

        self.pending = False

        self.galleries = galleries
        self.dynamic_gallery = (galleries is None)
        self.params = params
        self.files = files

        if (self.params.load_galeries == True):
            self.feature_galleries : List['self.ReIDGallery'] = []
            home = os.path.expanduser('~')
            for i in range(self.params.targets):
                filetypes = (('PyTorch Tensor files', '*.pt'), ('All files', '*.*'))
                filenames = filedialog.askopenfilenames(
                    title=f'Select gallery {i}',
                    initialdir=f'{home}/PDDA_Extracts/',
                    filetypes=filetypes)
                gal = list()
                for fn in filenames:
                    t = torch.load(fn)
                    gal.append(t)
                self.feature_galleries.append(gal)
            self.dynamic_gallery = False
            self.params.dynamic_gallery_content = False


        if (self.dynamic_gallery == True):
            self.feature_galleries = deque()

        
        rospy.loginfo("OSNET network initializing!")
        self.extractor = FeatureExtractor(
            model_name=self.files.model, model_path=self.files.getPath(), device=device)
        rospy.loginfo("OSNET network initialized succesfully.")

    def callback(self, msg : WindowPack):
        if self.pending:
            return
        # rospy.logdebug("ReID received message")
        self.pending = True
        self.wp = msg
        pass

    def detect(self):

        img = bridge.compressed_imgmsg_to_cv2(self.wp.img, "bgr8")
        height, width, channels = img.shape
        
        window_scores = []

        if (self.params is not None):
            if (self.params.redo_windows):
                for pw in self.wp.data:
                    # Consider desired aspect ratio 1:2 (w:h)
                    diff = pw.window.h - pw.window.w*self.params.window_ratio
                    if diff > 0:
                        pw.window.w = int(pw.window.h / self.params.window_ratio)
                    elif diff < 0:
                        pw.window.h = int(pw.window.w * self.params.window_ratio)
        
        descriptor = []
        try:
            for pw in self.wp.data:
                xLeft = int(max(0, pw.window.x - pw.window.w/2))
                yUp = int(max(0, pw.window.y - pw.window.h/2))
                xRight = int(min(width, pw.window.x + pw.window.w/2 - 1))
                yDown = int(min(height, pw.window.y + pw.window.h/2 - 1))

                cropped = img[yUp:yDown, xLeft:xRight]
                cropped = imutils.resize(cropped, width = self.params.window_w, height = int(self.params.window_w*self.params.window_ratio))
                descriptor.append(self.extractor(cropped))
                gallery_score = []
                for gallery in self.feature_galleries:
                    scores = []
                    for element in gallery:
                        scores.append(distance.cosine(descriptor[len(descriptor)-1], element))
                    gallery_score.append(numpy.min(scores))
                window_scores.append(gallery_score)

            window_scores = numpy.matrix(window_scores)
            
            row_index, col_index = linear_sum_assignment(window_scores)

            for i in range(len(row_index)):
                if window_scores[row_index[i], col_index[i]] > self.params.max_distance and self.dynamic_gallery:
                    continue
                self.wp.data[row_index[i]].assignment = col_index[i] + 1

                if self.params.dynamic_gallery_content:
                    if len(self.feature_galleries[col_index[i]]) >= self.params.descriptor_size:
                        self.feature_galleries[col_index[i]].popleft()
                if len(self.feature_galleries[col_index[i]]) < self.params.descriptor_size and self.params.load_galeries == False:
                    self.feature_galleries[col_index[i]].append(descriptor[row_index[i]])

            
            if (self.dynamic_gallery):
                for index in range(len(self.wp.data)):
                    if len(self.feature_galleries) < self.params.targets:
                        if (self.wp.data[index].assignment == 0):
                            gal = deque()
                            gal.append(descriptor[index])
                            self.feature_galleries.append(gal)
                            self.wp.data[index].assignment = len(self.feature_galleries)
                        
        except Exception as e:
            rospy.logwarn(f'{e}')
            
        self.wp.timestamp = rospy.Time.now()
        self.dataP.publish(self.wp)
        # rospy.logdebug("ReID sent data")
        self.pending = False
        

#   Functions

def reid():

    try:
        files = ReID.ReIDFiles(rospy.get_param("cfg/reid/path"), rospy.get_param("cfg/reid/model"))
        params = ReID.ReIDParameters(
            rospy.get_param("cfg/reid/descriptor_size"), 
            rospy.get_param("cfg/reid/window_w"), 
            rospy.get_param("cfg/reid/window_ratio"), 
            rospy.get_param("cfg/reid/max_distance"), 
            rospy.get_param("cfg/reid/dynamic_gallery"), 
            rospy.get_param("/master/target_no"), 
            rospy.get_param("cfg/reid/redo_yolo_windows"),
            rospy.get_param('cfg/reid/load_galleries')
            )
    except Exception:
        rospy.logerr("Configuration info missing! Load configuration file / Run configuration script, then try again!")
        rospy.signal_shutdown("Node cannot run without configuration info!")
        return
    
    obj = ReID(files, params)
    rospy.on_shutdown(world_end)

    rospy.loginfo("Robot ReID(OSNET) node running!")

    while not rospy.is_shutdown():
        if (obj.pending == True):
            obj.detect()

def world_end():
    rospy.loginfo("Robot ReID(OSNET) shutting down.")


#   Guard

if __name__ == '__main__':    
    
    # Initialize node
    rospy.init_node("reid", anonymous=False, log_level=rospy.DEBUG)
    rospy.loginfo("Robot ReID(OSNET) Node initializing...")

    # Launch in execution
    try:
        reid()
    except rospy.ROSInterruptException:
        rospy.logerr("Robot ReID(OSNET) Node initialization error!")
        pass
