from time import sleep

from text_detection import east_text_detect_model

DEFAULT_IMG_SAVE_DIR = 'capture'

class image_processor:
    def __init__(self, 
                 image_preprocessor = None, text_box_detector = None,
                 image_extracter = None, ocr_provider = None,
                 **kwargs) -> None:
        
        if image_preprocessor is not None:
            self.image_preprocessor = image_preprocessor
        else:
            self.image_preprocessor = None

        if text_box_detector is not None:
            self.text_box_detector = text_box_detector
        else:
            self.text_box_detector = east_text_detect_model(**kwargs)

        if image_extracter is not None:
            self.image_extracter = image_extracter
        else:
            self.image_extracter = None
        
        if ocr_provider is not None:
            self.ocr_provider = ocr_provider
        else:
            self.ocr_provider = None
        
        self.image_preprocess = lambda img : {'default': img}
        self.text_box_detect = lambda img: self.text_box_detector.detect(img)
        self.extract_postprocess_image = lambda img, box: self.image_extracter.extract_image(img, box)
        self.gen_ocr_report = lambda img_list: self.ocr_provider.recognize(img_list)

    def process(self, original_img):
        # Step 1 - do any image preprocessing
        pre_processed_imgs = self.image_preprocess(original_img)
        
        # Step 2 - do text detection
        report = {}
        for label, img in pre_processed_imgs.items():
            text_boxes = self.text_box_detect(img)
            
            if len(text_boxes) == 0:
                return
            
            label_report = {
                'img': img, 
                'text_boxes': text_boxes, 
                'extracted': [],
                'ocr': []
            }
            
            for box in label_report['text_boxes'].items():
                # Step 3 - draw a box around each test area
                if self.text_highlighter:
                    self.text_highlighter.highlight(img, box)
                
                extracted = self.extract(img, box)
                ocr = self.recognize(extracted)
                
                label_report['extracted'].append(extracted)
                label_report['ocr'].append(ocr)
                
            report[label] = label_report
            
        return report ## {label1: {img: {}, text_boxes: [], extracted: [], text: []}}
    
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