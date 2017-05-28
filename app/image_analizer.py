import cv2
import numpy as np
import os

from moments import HuInvariants
from image_modifier import ImageModifier
from image_segmenter import ImageSegmenter

class ImageAnalizer:
    def analize(image_path):
        print("Analizing... " + image_path)
        image = cv2.imread(image_path, 3)

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        way = np.where((image_hsv > 23) & (image_hsv < 36), 255, 0)[:,:,0]

        kernel = np.ones((5,5),np.uint8)

        way = ImageModifier().erode(way)
        way = ImageModifier().dilate(way)

        cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_way.png'), way);

        way_segments, boxes = ImageSegmenter(way).find_segments()
        cv2.imwrite(os.path.join(os.path.dirname(image_path), '2_way_segments.png'), way_segments);
