class ImageUtils:
    def surrounding(self, image, index):
        shape = image.shape

        left = (index[0], index[1] - 1) if (index[1] > 0) else None
        right = (index[0], index[1] + 1) if (index[1] + 1 < shape[1]) else None

        top = (index[0] - 1, index[1]) if (index[0] > 0) else None
        bottom = (index[0] + 1, index[1]) if (index[0] + 1 < shape[0]) else None

        directions = [left, top, right, bottom]
        return [d for d in directions if d is not None]
