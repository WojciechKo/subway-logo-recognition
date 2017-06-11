import cv2
import numpy as np
import os

from image_modifier import ImageModifier
from image_segmenter import ImageSegmenter
from moments import HuInvariants
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

        print("Labeling...")
        labeler = SegmentLabeler(self.model)

        labels = defaultdict(list)
        for box, segment in image_segmenter.segments():
            label, distance = labeler.label(segment)
            if label == None:
                continue
            labels[label].append(Segment(box, segment))
            file_name = "4_label_" + str(int(distance*1000000)) + "_" + str(label) + ".png"
            cv2.imwrite(os.path.join(os.path.dirname(image_path), file_name), segment);
        print(dict(labels))

        print("Finding whole logo...")
        group_by_labels = tuple(labels.values())
        word_combinations = cartesian(group_by_labels)
        subways = SubwayFinder().find_subways(word_combinations)

        for subway in subways:
            y = reduce(np.union1d, [segment.box[0] for segment in subway])
            x = reduce(np.union1d, [segment.box[1] for segment in subway])
            marker = (100, 100, 100)
            import pdb; pdb.set_trace()
            image[min(y):max(y) , min(x)] = marker
            image[max(y) , min(x):max(x) + 1] = marker
            image[min(y), min(x):max(x)] = marker
            image[min(y):max(y) , max(x)] = marker

        cv2.imwrite(os.path.join(os.path.dirname(image_path), "4_markers.png"), image);


class SubwayFinder:
    def find_subways(self, combinations):
        result = []

        comb_with_errors = [(comb, self.error_value(comb)) for comb in combinations]
        print("Combinations: " + str(comb_with_errors))
        best = self.best_fitted_combination(comb_with_errors)
        print("Next best: " + str(best))

        while (best != None):
            result.append(best)
            comb_with_errors = self.remove_combination(comb_with_errors, best)
            print("Combinations: " + str(comb_with_errors))

            best = self.best_fitted_combination(comb_with_errors)
            print("Next best: " + str(best))

        return result

    def best_fitted_combination(self, comb_with_errors):
        best = None
        min_error = math.inf

        for comb, error in comb_with_errors:
            if error < min_error:
                min_error = error
                best = comb

        return best

    def error_value(self, comb):
        x = [np.mean(segment.box[0]) for segment in comb]
        y = [np.mean(segment.box[1]) for segment in comb]

        a, b = np.polyfit(x, y, 1)
        prediction = np.vectorize(lambda x: a*x + b)

        error = np.mean((prediction(x) - y) ** 2)
        print("Mean squared error: %.2f" % error)

        return error

    def remove_combination(self, combinations, best):
        rest_combinations = []
        for combination, error in combinations:
            if combination[0] != best[0] and combination[1] != best[1] and  combination[2] != best[2] and  combination[3] != best[3] and combination[4] != best[4] and combination[5] != best[5]:
                rest_combinations.append((combination, error))
        return rest_combinations

class Segment:
    def __init__(self, box, segment):
        self.box = box
        self.segment = segment

class SegmentLabeler:
    def __init__(self, model):
        self.model = model
        self.scores = { letter: self.normalize(invariants) for letter, invariants in self.model.invariants.items() }

    def normalize(self, invariants):
        return { invariant_degree: self.z_score( value, invariant_degree) for invariant_degree, value in invariants.items() }

    def label(self, image):
        counts = np.bincount(image.flatten())
        fillness = counts[255] / (counts[0] + counts[255])
        if fillness > 0.60:
            return (None, 0)

        segment_invariants = HuInvariants(image).invariants()
        distances = { letter: self.distance(letter, segment_invariants) for letter, _invariants in self.model.invariants.items() }

        best_fit_letter = min(distances, key=lambda key: distances[key])        
        best = {k: v for k, v in distances.items() if k == best_fit_letter}
        return list(best.items())[0]

    def distance(self, letter, invariants):
        base = self.model.invariants[letter]
        normalized = self.normalize(invariants)

        select = lambda invariants: [invariants[degree] for degree in [3, 4, 7]]
        np.set_printoptions(suppress=True)
        dist = np.sum(np.absolute(np.subtract(select(self.scores[letter]), select(normalized))))

        return dist

    def z_score(self, value, invariant_degree):
        invariants = [values[invariant_degree] for values in self.model.invariants.values()]
        min_i = min(invariants)
        max_i = max(invariants)

        return (value - min_i) / (max_i - min_i)

