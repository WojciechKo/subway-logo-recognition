import numpy as np
from image_utils import ImageUtils

class ImageModifier:
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
