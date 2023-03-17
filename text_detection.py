from threading import Thread, Lock
from time import sleep
import numpy as np
import cv2
from pyimagesearch.east import EAST_OUTPUT_LAYERS
from pyimagesearch.east import decode_predictions

class east_text_detect_model:
    def __init__(self, **kwargs):
        # predefined model presets
        # construct a blob from the image, using model training params
        self.model_preset_means = (123.68, 116.78, 103.94)
        self.model_cnn_size = (None, None)
        self.minConf=kwargs["min_conf"]
        self.min_conf= kwargs["min_conf"]
        self.nms_thresh = kwargs["nms_thresh"]
        
        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        self.net = cv2.dnn.readNet(kwargs["east"])

        # check if we are going to use GPU
        if kwargs["use_gpu"]:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            # otherwise we are using our CPU
            print("[INFO] using CPU for inference...")

    def detect(self, img):
        # create blob from image
        blob = cv2.dnn.blobFromImage(
            img, 1.0, self.model_cnn_size,
            self.model_preset_means, 
            swapRB=True, crop=False
        )
        
        # set model input using blob
        self.net.setInput(blob)
        
        # and then perform a forward pass
        # of the model to obtain the two output layer sets
        (scores, geometry) = self.net.forward(EAST_OUTPUT_LAYERS)

        # decode the predictions form OpenCV's EAST text detector 
        (rects, confidences) = decode_predictions(scores, geometry,minConf=self.min_conf)
        
        # apply non-maximum suppression (NMS) to the rotatedbounding boxes
        text_boxes = cv2.dnn.NMSBoxesRotated(rects, confidences,self.min_conf, self.nms_thresh)
        
        return text_boxes