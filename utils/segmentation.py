import numpy as np
import matplotlib.pyplot as plt


def get_coupling_medium_segmentation(p0):
    segmentation_mask = np.zeros_like(p0).astype(bool)

    center = (150, 150)
    radius = 125
    y, x = np.ogrid[:300, :300]
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2)

    threshold = np.median(p0[~mask])
    segmentation_mask[~mask] = 1
    segmentation_mask[mask] = p0[mask] > threshold

    return segmentation_mask