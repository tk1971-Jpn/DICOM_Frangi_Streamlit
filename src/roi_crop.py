import numpy as np


def clamp_roi(z1, z2, y1, y2, x1, x2, shape):
    zmax, ymax, xmax = shape

    z1 = max(0, min(z1, zmax - 1))
    z2 = max(z1 + 1, min(z2, zmax))

    y1 = max(0, min(y1, ymax - 1))
    y2 = max(y1 + 1, min(y2, ymax))

    x1 = max(0, min(x1, xmax - 1))
    x2 = max(x1 + 1, min(x2, xmax))

    return z1, z2, y1, y2, x1, x2


def crop_volume(volume: np.ndarray, z1, z2, y1, y2, x1, x2):
    return volume[z1:z2, y1:y2, x1:x2]