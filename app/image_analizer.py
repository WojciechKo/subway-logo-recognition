import numpy as np
import cv2
import os
import datetime

from image_processing import ImageThresholder, ImageModifier, ImageSegmenter, LetterClassificator, SubwayFinder

class ImageAnalizer:
    def __init__(self, model):
        self.model = model
        self._now = datetime.datetime.now()

    def analize(self, image_path):      
        folder_path = os.path.dirname(image_path)
        print("\nAnalizing... " + folder_path)
        image = cv2.imread(image_path, 3)

        self._check_time(False)

        print("Thresholding...", end='', flush=True)
        subway_image = ImageThresholder().extract_subway(image)
        cv2.imwrite(os.path.join(folder_path, '1_subway.png'), subway_image)
        self._check_time()

        print("Erode..." , end='', flush=True)
        subway_image = ImageModifier().erode(subway_image, 3)
        self._check_time()

        print("Dilate..." , end='', flush=True)
        subway_image = ImageModifier().dilate(subway_image, 3)

        cv2.imwrite(os.path.join(folder_path, '2_subway.png'), subway_image)
        # subway_image = cv2.imread(os.path.join(folder_path, '2_subway.png'), 0)

        self._check_time()

        print("Segmentation..." , end='', flush=True)
        image_segmenter = ImageSegmenter(subway_image)
        cv2.imwrite(os.path.join(folder_path, '3_segments.png'), image_segmenter.segmented_image);
        self._check_time()

        print("Letter classification..." , end='', flush=True)
        classificator = LetterClassificator(self.model)
        grouped_segments = classificator.group_segments(image_segmenter.segments)
        self._check_time()

        for _letter, classifications in grouped_segments.items():
            for classification in classifications:
                file_name = "4_label_" + str(classification.letter) + "_" + str(int(classification.distance * 1000)) + "_" + str(id(classification.segment))+ ".png"
                cv2.imwrite(os.path.join(folder_path, file_name), classification.segment.image);

        self._check_time(False)

        print("Finding whole logo..." , end='', flush=True)
        subways = SubwayFinder().find_subways(grouped_segments)

        for subway in subways:
            ImageModifier().highlight_classifications(image, subway)

        cv2.imwrite(os.path.join(folder_path, "4_markers.png"), image);
        self._check_time()

    def _check_time(self, display = True):
        new_now = datetime.datetime.now()
        delta = new_now - self._now
        if display:
            print(' took %5.3fs.' % delta.total_seconds())
        self._now = new_now
        return delta
