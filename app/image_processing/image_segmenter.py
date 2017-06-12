import numpy as np

from .image_utils import ImageUtils

class Segment:
    def __init__(self, box, image):
        self.box = box
        self.image = image

class ImageSegmenter:
    def __init__(self, image):
        self._image = image
        self.segments = self.find_segments()

    @property
    def segmented_image(self):
        return self._segmented_image

    def find_segments(self):
        self._segmented_image = np.zeros_like(self._image)

        group = 1
        segments = []

        it = np.nditer(self._image, flags=['multi_index'])
        while not it.finished:
            if it[0] == 255 and self._segmented_image[it.multi_index] == 0:
                segment = self.find_segment(it.multi_index, group)
                segments.append(segment)
                group += 1
            it.iternext()

        return segments

    def find_segment(self, start_point, group):
        points = set([start_point])
        min_y, min_x = start_point
        max_y, max_x = start_point

        while(len(points) > 0):
            point = points.pop()
            self._segmented_image[point] = group
            for surround in ImageUtils().neighbourhood(self._image, point, 1):
                if self._image[surround] == 255 and self._segmented_image[surround] == 0:
                    sur_y, sur_x = surround

                    min_x = min(min_x, sur_x)
                    max_x = max(max_x, sur_x)
                    min_y = min(min_y, sur_y)
                    max_y = max(max_y, sur_y)
                    points.add(surround)

        box = (np.arange(min_y, max_y), np.arange(min_x, max_x))
        image = np.where(self._segmented_image == group, 255, 0)[box[0].min():box[0].max(), box[1].min():box[1].max()]

        return Segment(box, image)
