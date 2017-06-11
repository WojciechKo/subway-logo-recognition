import numpy as np
import cv2
import os
import itertools
import datetime

from collections import defaultdict
from functools import reduce
from image_processing import ImageModifier, ImageSegmenter, LetterClassificator, SubwayFinder

class ImageAnalizer:
    def __init__(self, model):
        self.model = model
        self._now = datetime.datetime.now()

    def analize(self, image_path):      
        print("\nAnalizing... " + image_path)

        image = cv2.imread(image_path, 3)

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = image_hsv[:,:,0]
        s = image_hsv[:,:,1]
        v = image_hsv[:,:,2]

        way_image = np.where((h > 20) & (h < 36) & (v > 200), 255, 0)
        sub_image = np.where((s < 30) & (v > 220), 255, 0)

        cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_sub.png'), sub_image)
        cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_way.png'), way_image)

        subway_image = np.maximum(way_image, sub_image)
        self._check_time(False)

        print("Erode..." , end='', flush=True)
        subway_image = ImageModifier().erode(subway_image, 3)
        self._check_time()

        print("Dilate..." , end='', flush=True)
        subway_image = ImageModifier().dilate(subway_image, 3)

        cv2.imwrite(os.path.join(os.path.dirname(image_path), '2_subway.png'), subway_image)
        # subway_image = cv2.imread(os.path.join(os.path.dirname(image_path), '2_subway.png'), 0)

        self._check_time()

        print("Segmentation..." , end='', flush=True)
        image_segmenter = ImageSegmenter(subway_image)
        cv2.imwrite(os.path.join(os.path.dirname(image_path), '3_segments.png'), image_segmenter.segmented_image);
        self._check_time()

        print("Letter classification..." , end='', flush=True)
        classificator = LetterClassificator(self.model)

        grouped_segments = defaultdict(list)
        for segment in image_segmenter.segments:
            letter, distance = classificator.classify(segment.image)
            if letter == None: continue

            grouped_segments[letter].append(segment)
            file_name = "4_label_" + str(letter) + "_" + str(int(distance * 100)) + ".png"
            cv2.imwrite(os.path.join(os.path.dirname(image_path), file_name), segment.image);

        self._check_time()

        if not all(letter in grouped_segments for letter in ("S","U","B","W","A","Y")):
            print("Can not find all parts of logo")
            return

        self._check_time(False)
        print("Finding whole logo..." , end='', flush=True)
        sorted_groups = [grouped_segments[letter] for letter in ("S","U","B","W","A","Y")]
        logo_propositions = list(itertools.product(*sorted_groups))
        subways = SubwayFinder().find_subways(logo_propositions)

        for subway in subways:
            marker = (50, 50, 255)
            for segment in subway:
                image[segment.box[0].min(): segment.box[0].max(), segment.box[1].min():segment.box[1].max()] = marker

            y = reduce(np.union1d, [segment.box[0] for segment in subway])
            x = reduce(np.union1d, [segment.box[1] for segment in subway])

            image[min(y):max(y) , min(x)] = marker
            image[max(y), min(x):max(x) + 1] = marker
            image[min(y), min(x):max(x)] = marker
            image[min(y):max(y) , max(x)] = marker

        cv2.imwrite(os.path.join(os.path.dirname(image_path), "4_markers.png"), image);
        self._check_time()

    def _check_time(self, display = True):
        new_now = datetime.datetime.now()
        delta = new_now - self._now
        if display:
            print(' took %5.3fs.' % delta.total_seconds())
        self._now = new_now
        return delta
