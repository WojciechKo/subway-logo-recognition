import os
import yaml
import numpy as np
import cv2
import logging

from image_processing import HuInvariants, ImageModifier, ImageSegmenter

class SubwayInvariantsModel:
    def __init__(self, image_path):
        try:
            with open(os.path.join(os.path.dirname(image_path), "invariants.yml"), 'r') as stream:
                self._invariants = yaml.load(stream)
        except FileNotFoundError:
            self._invariants = self.calculate(image_path)
        except yaml.YAMLError as exc:
            self._invariants = self.calculate(image_path)
    @property
    def invariants(self):
        return self._invariants

    def calculate(self, image_path):
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        subway = np.where(image_gray > 180, 255, 0)

        i = 1
        mapper = {1: "Y", 2: "S", 3: "W", 4: "A", 5: "B", 6: "U"}
        results = {}

        logger = logging.getLogger()
        logger.info("Erode...")
        subway = ImageModifier().erode(subway, 3)
        logger.info("Dilate...")
        subway = ImageModifier().dilate(subway, 3)

        for segment in ImageSegmenter(subway).segments:
            if i > 6: break

            cv2.imwrite(os.path.join(os.path.dirname(image_path), "way_segmented_{0}.png".format(mapper[i])), segment.image);
            results[mapper[i]] = HuInvariants(segment.image).invariants()

            i += 1

        with open(os.path.join(os.path.dirname(image_path), "invariants.yml"), 'w') as outfile:
            yaml.dump(results, outfile, default_flow_style=False)

        return results
