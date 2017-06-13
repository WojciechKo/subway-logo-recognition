import numpy as np
import cv2
import os
import datetime
import logging


from image_processing import ImageThresholder, ImageModifier, ImageSegmenter, LetterClassificator, SubwayFinder

class ImageAnalizer:
    def __init__(self, model):
        self.model = model
        self._now = datetime.datetime.now()

    def analize(self, image_path):      
        folder_path = os.path.dirname(image_path)
        logger = logging.getLogger()

        logger.info("Analizing... " + folder_path)

        image = cv2.imread(image_path, 3)

        self._check_time()
        subway_image = ImageThresholder().extract_subway(image)
        logger.info("Thresholding... " + str(self._check_time()))

        cv2.imwrite(os.path.join(folder_path, '1_subway.png'), subway_image)

        self._check_time()
        subway_image = ImageModifier().erode(subway_image, 3)
        logger.info("Erode... " + str(self._check_time()))

        self._check_time()
        subway_image = ImageModifier().dilate(subway_image, 3)
        logger.info("Dilate... " + str(self._check_time()))

        cv2.imwrite(os.path.join(folder_path, '2_subway.png'), subway_image)
        # subway_image = cv2.imread(os.path.join(folder_path, '2_subway.png'), 0)

        self._check_time()
        image_segmenter = ImageSegmenter(subway_image)
        logger.info("Segmentation... " + str(self._check_time()))
        cv2.imwrite(os.path.join(folder_path, '3_segments.png'), image_segmenter.segmented_image);

        self._check_time()
        classificator = LetterClassificator(self.model)
        grouped_segments = classificator.group_segments(image_segmenter.segments)
        logger.info("Letter classification..." + str(self._check_time()))

        for _letter, classifications in grouped_segments.items():
            for classification in classifications:
                file_name = "4_label_" + str(classification.letter) + "_" + str(int(classification.distance * 1000)) + "_" + str(id(classification.segment))+ ".png"
                cv2.imwrite(os.path.join(folder_path, file_name), classification.segment.image);

        self._check_time()
        subways = SubwayFinder().find_subways(grouped_segments)
        logger.info("Finding whole logo..." + str(self._check_time()))

        markers_image = ImageModifier().highlight_logos(image, subways)
        markers_debug_image = ImageModifier().highlight_logos(image, subways, debug=True)

        cv2.imwrite(os.path.join(folder_path, "4_markers.png"), markers_image);
        cv2.imwrite(os.path.join(folder_path, "4_markers_debug.png"), markers_debug_image);

    def _check_time(self):
        new_now = datetime.datetime.now()
        delta = new_now - self._now
        self._now = new_now
        return ' took %5.3fs.' % delta.total_seconds()
