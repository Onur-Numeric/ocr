import argparse
from threading import Thread, Lock
import cv2

from process_frame import process

keep_going = True
frame = None
frame_mutex = Lock()

show_preview = True

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
    
    if show_preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        
    while keep_going:
        if frame is not None:
            frame_mutex.acquire()
            frame_snapshot = frame.copy() # copy the current frame if required
            frame_mutex.release()
            
            if show_preview:
                cv2.imshow('preview', frame_snapshot)
            
            # TODO - do the text detect magic here
            process(frame_snapshot)
             
            if cv2.waitKey(1) == ord('q'):
                keep_going = False
                break
     
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-p", "--model_path", required=True,
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