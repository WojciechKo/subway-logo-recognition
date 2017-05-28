import cv2
import numpy as np
import os
import yaml

from moments import HuInvariants
from image_segmenter import ImageSegmenter

class InvariantsCalculator:
    def calculate(self, image_path):
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        way = np.where(image_gray > 180, 255, 0)

        i = 1
        mapper = {1: "Y", 2: "S", 3: "W", 4: "A", 5: "B", 6: "U"}

        segments = ImageSegmenter(way).segments()
        results = {}

        for image in segments:
            if i > 6: break 

            cv2.imwrite(os.path.join(os.path.dirname(image_path), "way_segmented_{0}.png".format(mapper[i])), image);

            results[mapper[i]] = HuInvariants(image).invariants()

            i += 1

        with open(os.path.join(os.path.dirname(image_path), "invariants.yml"), 'w') as outfile:
            yaml.dump(results, outfile, default_flow_style=False)

        return results
