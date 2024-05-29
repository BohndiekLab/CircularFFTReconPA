from utils.load_data import load_all, load_all_recons
from scipy.stats import linregress
import numpy as np
from utils.segmentation import get_coupling_medium_segmentation, get_unit_circle
import os
import inspect
import matplotlib.pyplot as plt


def calibrate_to_p0(algorithm, data_source):

    cal_file = f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}/cal_{algorithm}_{data_source}.npz"

    if os.path.exists(cal_file):
        cal_data = np.load(cal_file)
        return cal_data["slope"], cal_data["intercept"], cal_data["r"]

    p0 = load_all("calibration", "p0")
    target = load_all_recons("calibration", data_source, algorithm)

    for idx in range(len(p0)):
        segmentation_mask = get_coupling_medium_segmentation(p0[idx])
        p0[idx][~segmentation_mask] = np.nan
        target[idx][~segmentation_mask] = np.nan

    p0 = np.reshape(p0, (-1, ))
    target = np.reshape(target, (-1,))
    print(len(p0))
    p0 = p0[~np.isnan(p0)]
    target = target[~np.isnan(target)]
    print(len(p0))

    slope, intercept, r, p, _ = linregress(target, p0)

    print(slope, intercept, r, p)

    np.savez(cal_file,
             slope=slope,
             intercept=intercept,
             r=r,
             p=p)

    return slope, intercept, r
