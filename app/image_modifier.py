import numpy as np
from image_utils import ImageUtils

class ImageModifier:
    def erode(self, image):
        it = np.nditer(image, flags=['multi_index'])
        result = np.zeros_like(image)

        while not it.finished:
            if it[0] == 255:
                surrounds = ImageUtils().surrounding(image, it.multi_index)
                if len([s for s in surrounds if image[s] == 255]) == 4:
                    result[it.multi_index] = 255
            it.iternext()

        return result

    def dilate(self, image):
        it = np.nditer(image, flags=['multi_index'])
        result = np.zeros_like(image)

        while not it.finished:
            if it[0] == 255:
                for s in ImageUtils().surrounding(image, it.multi_index):
                    result[s] = 255
            it.iternext()

        return result
