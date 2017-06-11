import cv2
import numpy as np
import os

from image_modifier import ImageModifier
from image_segmenter import ImageSegmenter
from moments import HuInvariants
from collections import defaultdict
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

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = image_hsv[:,:,0]
        s = image_hsv[:,:,1]
        v = image_hsv[:,:,2]

        way_image = np.where((h > 20) & (h < 36) & (v > 200), 255, 0)
        sub_image = np.where((s < 30) & (v > 220), 255, 0)

        cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_sub.png'), sub_image)
        cv2.imwrite(os.path.join(os.path.dirname(image_path), '1_way.png'), way_image)

        subway_image = np.maximum(way_image, sub_image)

        print("Erode...")
        subway_image = ImageModifier().erode(subway_image, 3)
        print("Dilate...")
        subway_image = ImageModifier().dilate(subway_image, 3)

        cv2.imwrite(os.path.join(os.path.dirname(image_path), '2_subway.png'), subway_image)
        # subway_image = cv2.imread(os.path.join(os.path.dirname(image_path), '2_subway.png'), 0)

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
            file_name = "3_label_" + str(int(distance*1000000)) + "_" + str(label) + ".png"
            cv2.imwrite(os.path.join(os.path.dirname(image_path), file_name), segment);
        print(dict(labels))
        import pdb; pdb.set_trace()

        group_by_labels = tuple(labels.values)
        word_combinations = cartesian(group_by_labels)
        print(word_combinations)
        return labels

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
        if fillness > 0.65:
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

