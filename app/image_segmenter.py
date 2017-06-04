import numpy as np
from image_utils import ImageUtils

class ImageSegmenter:
    def __init__(self, image):
        self.image = image
        self.boxes = {}

        self.find_segments()

    def segments(self):
        return [(self.boxes[group_id], self.segment(group_id)) for group_id in self.boxes]

    @property
    def segmented_image(self):
        return self._segmented_image
    
    # PRIVATE PART duno how :(
    def find_segments(self):
        self._segmented_image = np.zeros_like(self.image)
        self.boxes = {}

        group = 1
        it = np.nditer(self.image, flags=['multi_index'])
        while not it.finished:
            if it[0] == 255 and self._segmented_image[it.multi_index] == 0:
                self.boxes[group] = self.find_segment(it.multi_index, group)
                group += 1
            it.iternext()

        return (self._segmented_image, self.boxes)

    def segment(self, key):
        box = self.boxes[key]
        return np.where(self._segmented_image == key, 255, 0)[box[0].min():box[0].max(), box[1].min():box[1].max()]

    def find_segment(self, index, group):
        points = set([index])
        min_y, min_x = index
        max_y, max_x = index

        while(len(points) > 0):
            point = points.pop()
            self._segmented_image[point] = group
            for surround in ImageUtils().neighbourhood(self.image, point, 1):
                if self.image[surround] == 255 and self._segmented_image[surround] == 0:
                    sur_y, sur_x = surround

                    min_x = min(min_x, sur_x)
                    max_x = max(max_x, sur_x)
                    min_y = min(min_y, sur_y)
                    max_y = max(max_y, sur_y)
                    points.add(surround)

        return (np.arange(min_y, max_y), np.arange(min_x, max_x))



