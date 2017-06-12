import numpy as np

from functools import reduce

from .image_utils import ImageUtils

class ImageModifier:
    def highlight_classifications(self, image, classifications, marker = (50, 50, 255)):
        segments = [classification.segment for classification in classifications]

        for segment in segments:
            image[segment.box[0].min(): segment.box[0].max(), segment.box[1].min():segment.box[1].max()] = marker

        y = reduce(np.union1d, [segment.box[0] for segment in segments])
        min_y = min(y)
        max_y = max(y)

        x = reduce(np.union1d, [segment.box[1] for segment in segments])
        min_x = min(x)
        max_x = max(x)

        image[min_y:max_y , min_x] = marker
        image[max_y, min_x:max_x + 1] = marker
        image[min_y, min_x:max_x] = marker
        image[min_y:max_y , max_x] = marker

    def erode(self, image, size = 1):
        it = np.nditer(image, flags=['multi_index'])
        result = np.zeros_like(image)

        while not it.finished:
            if it[0] == 255:
                surrounds = ImageUtils().neighbourhood(image, it.multi_index, size)
                if np.all([image[s] for s in surrounds]):
                    result[it.multi_index] = 255
            it.iternext()

        return result

    def dilate(self, image, size = 1):
        it = np.nditer(image, flags=['multi_index'])
        result = np.zeros_like(image)

        while not it.finished:
            if it[0] == 255:
                for s in ImageUtils().neighbourhood(image, it.multi_index, size):
                    result[s] = 255
            it.iternext()

        return result

    def median_blur(self, image, size = 1):
        it = np.nditer(image, flags=['multi_index'])
        result = np.zeros_like(image)

        while not it.finished:
            result[it.multi_index] = np.median([image[s] for s in ImageUtils().surrounding(image, it.multi_index)])
        return result
