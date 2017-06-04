import numpy as np

class ImageUtils:
    def __init__(self):
        self.masks = {}

    def surrounding(self, image, index):
        shape = image.shape

        left = (index[0], index[1] - 1) if (index[1] > 0) else None
        right = (index[0], index[1] + 1) if (index[1] + 1 < shape[1]) else None

        top = (index[0] - 1, index[1]) if (index[0] > 0) else None
        bottom = (index[0] + 1, index[1]) if (index[0] + 1 < shape[0]) else None

        directions = [left, top, right, bottom]
        return [d for d in directions if d is not None]

    def neighbourhood(self, image, index, size):
        height, width = image.shape
        row, column = index

        row_min = max(row - size, 0)
        row_max = min(row + size, height - 1)

        column_min = max(column - size, 0)
        column_max = min(column + size, width - 1)

        arr = np.array([ (row_i, col_i) for row_i in range(row_min, row_max + 1) for col_i in range(column_min, column_max + 1) ])

        mask_row_min = row_min - (row - size)
        mask_row_max = 2*size + row_max - (row + size) + 1
        mask_column_min = column_min - (column - size)
        mask_column_max = 2*size + column_max - (column + size) + 1

        mask = self.mask(size)[mask_row_min : mask_row_max, mask_column_min : mask_column_max].flatten()

        return [(r,c) for r, c in arr[mask]]

    def mask(self, size):
        if not size in self.masks:
            diameter = size*2 + 1

            y,x = np.ogrid[-size:diameter-size, -size:diameter-size]
            mask = x*x + y*y <= size*size
            self.masks[size] = np.array(mask)

        return self.masks[size]
