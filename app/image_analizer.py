import cv2
import numpy as np
import os

from image_modifier import ImageModifier
from image_segmenter import ImageSegmenter
from letter_classificator import LetterClassificator
from collections import defaultdict
import math
from functools import reduce

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

class ImageAnalizer:
    def __init__(self, model):
        self.model = model

    def analize(self, image_path):
        print("Analizing... " + image_path)

        image = cv2.imread(image_path, 3)

        # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # h = image_hsv[:,:,0]
        # s = image_hsv[:,:,1]
        # v = image_hsv[:,:,2]

        # way_image = np.where((h > 20) & (h < 36) & (v > 200), 255, 0)
        # sub_image = np.where((s < 30) & (v > 220), 255, 0)

        # cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_sub.png'), sub_image)
        # cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_way.png'), way_image)

        # subway_image = np.maximum(way_image, sub_image)

        # print("Erode...")
        # subway_image = ImageModifier().erode(subway_image, 3)
        # print("Dilate...")
        # subway_image = ImageModifier().dilate(subway_image, 3)

        # cv2.imwrite(os.path.join(os.path.dirname(image_path), '2_subway.png'), subway_image)
        subway_image = cv2.imread(os.path.join(os.path.dirname(image_path), '2_subway.png'), 0)

        print("Segmentation...")
        image_segmenter = ImageSegmenter(subway_image)
        cv2.imwrite(os.path.join(os.path.dirname(image_path), '3_segments.png'), image_segmenter.segmented_image);

        print("Letter classification...")
        classificator = LetterClassificator(self.model)

        grouped_segments = defaultdict(list)
        for segment in image_segmenter.segments:
            letter, distance = classificator.classify(segment.image)
            if letter == None: continue

            grouped_segments[letter].append(segment)
            file_name = "4_label_" + str(letter) + "_" + str(int(distance * 100)) + ".png"
            cv2.imwrite(os.path.join(os.path.dirname(image_path), file_name), segment.image);

        print(dict(grouped_segments))

        if not all(letter in grouped_segments for letter in ("S","U","B","W","A","Y")):
            print("Can not find all parts of logo")
            return

        print("Finding whole logo...")
        logo_propositions = cartesian(tuple(grouped_segments.values()))
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

# import pdb; pdb.set_trace()

class SubwayFinder:
    def find_subways(self, combinations):
        result = []

        errors = [self._mean_squared_error(combination) for combination in combinations]
        comb_with_errors = sorted(zip(combinations, errors), key=lambda pair: pair[1])
        while(len(comb_with_errors) > 0):
            best, error = self.best_fitted_combination(comb_with_errors)

            if best == None: break

            print("Next best error: " + str(error))

            result.append(best)
            comb_with_errors = self.remove_combination(comb_with_errors, best)

        return result

    def best_fitted_combination(self, comb_with_errors):
        if len(comb_with_errors) < 1:
            return(None, math.inf)

        return comb_with_errors[0]

    def _mean_squared_error(self, segments):
        y = [np.mean(segment.box[0]) for segment in segments]
        x = [np.mean(segment.box[1]) for segment in segments]

        a, b = np.polyfit(x, y, 1)
        prediction = np.vectorize(lambda x: a*x + b)

        return np.mean((prediction(x) - y) ** 2)

    def remove_combination(self, combinations, best):
        rest_combinations = []
        for combination, error in combinations:
            if combination[0] != best[0] and combination[1] != best[1] and  combination[2] != best[2] and  combination[3] != best[3] and combination[4] != best[4] and combination[5] != best[5]:
                rest_combinations.append((combination, error))
        return rest_combinations
