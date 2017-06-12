import numpy as np
import cv2

class ImageThresholder:
    def extract_subway(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = image_hsv[:,:,0]
        s = image_hsv[:,:,1]
        v = image_hsv[:,:,2]

        way_image = np.where((h > 20) & (h < 36) & (v > 200), 255, 0)
        sub_image = np.where((s < 30) & (v > 220), 255, 0)

        return np.maximum(way_image, sub_image)
