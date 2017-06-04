import cv2
import numpy as np
import os

from moments import HuInvariants
from image_modifier import ImageModifier
from image_segmenter import ImageSegmenter

class ImageAnalizer:
    def analize(self, image_path):
        print("Analizing... " + image_path)
        image = cv2.imread(image_path, 3)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = image_hsv[:,:,0]
        s = image_hsv[:,:,1]
        v = image_hsv[:,:,2]

        way_image = np.where((h > 20) & (h < 36) & (v > 200), 255, 0)
        sub_image = np.where((s < 30) & (v > 220), 255, 0)

        cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_sub.png'), sub_image)
        cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_way.png'), way_image)

        image = np.maximum(way_image, sub_image)

        print("erode...")
        image = ImageModifier().erode(image, 3)
        print("dilate...")
        image = ImageModifier().dilate(image, 3)

        cv2.imwrite(os.path.join(os.path.dirname(image_path), '2_subway.png'), image)

        image_segmenter = ImageSegmenter(image)
        cv2.imwrite(os.path.join(os.path.dirname(image_path), '3_segments.png'), image_segmenter.segmented_image);
