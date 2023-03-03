from time import sleep
import numpy as np
import cv2
from PIL import Image as ImageConverter

from text_detection import east_text_detect_model

DEFAULT_IMG_SAVE_DIR = 'capture'

class image_processor:
    def __init__(self, **kwargs) -> None:
        self.text_box_detect = east_text_detect_model(kwargs)

    def process(self, original_img):
        # Step 1 - do any image preprocessing
        pre_processed_img = original_img
        
        # Step 2 - do text detection
        text_boxes = self.text_box_detect.text_detection(pre_processed_img)
        if len(text_boxes) == 0:
            return
        
        # Step 3 - whatever comes next
    
    '''
    # print("Text Detected"+ str(len(idxs)))
    # ensure that at least one text bounding box was found
    if len(idxs) > 0:
        # print("Text Detected")
        # loop over the valid bounding box indexes after applying NMS
        for i in idxs.flatten():
            # compute the four corners of the bounding box, scale the
            # coordinates based on the respective ratios, and then
            # convert the box to an integer NumPy array
            box = cv2.boxPoints(rects[i])

            box[:, 0] *= rW
            box[:, 1] *= rH
            box = np.int0(box)

            # draw a rotated bounding box around the text
            cv2.polylines(resizedFrame, [box], True, (0, 255, 0), 2)

            gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)

            try:
                image1 = gray[min([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])-20:max([box[0, 1], box[1, 1], box[2, 1], box[3, 1]])+20,
                                  min([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])-20:max([box[0, 0], box[1, 0], box[2, 0], box[3, 0]])+20]
                backtorgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
                im1 = ImageConverter.fromarray(backtorgb)
                im2 = ImageConverter.fromarray(backtorgb)
                kerasimages = [origFrame] # [keras_ocr.tools.read(img)for img in [ 'editted.jpg']]
                print("starting recognition")
                prediction_groups = pipeline.recognize(kerasimages)
                print("prediction shape="+ str(type(prediction_groups)))
                for pred_grp in prediction_groups:
                    for text, box in pred_grp:
                        print(text)                  
            except Exception as e:
                print(f"Keras Exception: ", e)
                '''