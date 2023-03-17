import numpy as np
from PIL import Image
from pathlib import Path
from process_frame import image_processor

TEST_IMAGE_BASE_DIR = "test_images"
TEST_IMAGE_NAME = "test.jpg"

base_path = Path(TEST_IMAGE_BASE_DIR)
image_path = Path(TEST_IMAGE_BASE_DIR, TEST_IMAGE_NAME)

options = {
    "min_conf": 0.5,
    "nms_thresh": 0.4,
    "model_path": "",
    "east": "./models/east/frozen_east_text_detection.pb",
    "use_gpu": False
}

object_extractor = image_processor(
    image_preprocessor = None, text_box_detector = None,
    image_extracter = None, ocr_provider = None,
    **options)

with Image.open(image_path) as img:
    print(object_extractor.process(np.array(img)))
    img.show()