import argparse
from threading import Thread, Lock
from time import sleep
import numpy as np
import cv2
import keras_ocr
from PIL import Image as ImageConverter
from pyimagesearch.east import EAST_OUTPUT_LAYERS
from pyimagesearch.east import decode_predictions

keep_going = True
frame = None
frame_mutex = Lock()

IMG_SAVE_DIR = 'capture'

pipeline = keras_ocr.pipeline.Pipeline()
net = None

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (None, None)
(rW, rH) = (None, None)


def text_detection(origFrame):
    global pipeline, net, W
    global H
    global newW
    global newH
    global rW,  rH
    # resize the frame
    resizedFrame = cv2.resize(origFrame, (1000, 750))

    # if our frame dimensions are None, we still need to compute the
    # ratio of old frame dimensions to new frame dimensions
    if W is None or H is None:
        (H, W) = resizedFrame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)
        print("newW="+str(newW))
        print("newH="+str(newH))

    # construct a blob from the image and then perform a forward pass
    # of the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(resizedFrame, 1.0, (newW, newH),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)

    # decode the predictions form OpenCV's EAST text detector and
    # then apply non-maximum suppression (NMS) to the rotated
    # bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry,
        minConf=args["min_conf"])
    idxs = cv2.dnn.NMSBoxesRotated(rects, confidences,
        args["min_conf"], args["nms_thresh"])
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

            # kerasimages=[keras_ocr.tools.read("your_file1.jpeg"),keras_ocr.tools.read("your_file2.jpeg")]
            # kerasimages=[keras_ocr.tools.read(img)for img in ['https://storage.googleapis.com/gcptutorials.com/examples/keras-ocr-img-1.jpg','https://storage.googleapis.com/gcptutorials.com/examples/keras-ocr-img-2.png']]

            # print(kerasimages)                 

            #cv2.imshow('image', backtorgb)

def frame_grabber(src):
    global frame
    global frame_mutex
    global keep_going
        
    # if a video path was not supplied, grab the reference to the webcam
    if src is None:
        print("[INFO] starting default video stream...")
        vs = cv2.VideoCapture(0)
        print("[INFO] started default video stream...")
    else:
        # otherwise, grab a reference to the video file
        print(f"[INFO] starting video stream: '{src}'")
        vs = cv2.VideoCapture(src)
    
    print(f"[INFO] Starting framegrabber loop")
    ctr = 0
    while keep_going and vs.isOpened():
        ctr += 1
        ret, _frame = vs.read()
        frame_mutex.acquire()
        #TODO update the frame
        frame = _frame
        frame_mutex.release()
        if ctr%300 == 0:
            print(f"frame# {ctr}")
        if cv2.waitKey(1) == ord('q'):
            keep_going = False
            break

def frame_processor():
    global frame
    global frame_mutex
    global keep_going
    
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    while keep_going:
        if frame is not None:
            frame_mutex.acquire()
            frame_snapshot = frame.copy() # copy the current frame if required
            frame_mutex.release()
            
            # TODO - do the text detect magic here
            text_detection(frame_snapshot)
            cv2.imshow('preview', frame_snapshot)
            if cv2.waitKey(1) == ord('q'):
                keep_going = False
                break
     
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-e", "--east", required=True,
        help="path to input EAST text detector")
    ap.add_argument("-w", "--width", type=int, default=320,
        help="resized image width (should be multiple of 32)")
    ap.add_argument("-t", "--height", type=int, default=320,
        help="resized image height (should be multiple of 32)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.5,
        help="minimum probability required to inspect a text region")
    ap.add_argument("-n", "--nms-thresh", type=float, default=0.4,
        help="non-maximum suppression threshold")
    ap.add_argument("-g", "--use-gpu", type=bool, default=False,
        help="boolean indicating if CUDA GPU should be used")
    args = vars(ap.parse_args())

    # initialize the original frame dimensions, new frame dimensions,
    # and ratio between the dimensions
    (newW, newH) = (args["width"], args["height"])

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    # check if we are going to use GPU
    if args["use_gpu"]:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        # otherwise we are using our CPU
        print("[INFO] using CPU for inference...")
        
    src = None if args.get("input", False) else args["input"]
        
    grabber_th = Thread(target=frame_grabber, kwargs={"src": src})
    processor_th = Thread(target=frame_processor, args=())
    
    grabber_th.start()
    processor_th.start()
    
    grabber_th.join()
    processor_th.join()
    
    print("exiting..")