from scipy.stats import linregress
import numpy as np
from utils.segmentation import get_coupling_medium_segmentation, get_unit_circle
from utils.constants import get_recon_path, get_p0_path
import os
import inspect
import glob
import matplotlib.pyplot as plt


def calibrate_to_p0(algorithm, data_source):

    cal_file = f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}/cal_{algorithm}_{data_source}.npz"

    if os.path.exists(cal_file):
        cal_data = np.load(cal_file)
        return cal_data["slope"], cal_data["intercept"], cal_data["r"]

    p0_path = get_p0_path("calibration")
    p0_files = glob.glob(f"{p0_path}/*.npy")
    gt = []
    for p0_file in p0_files:
        gt.append(np.load(p0_file))
    p0 = np.asarray(gt)

    recon_path = get_recon_path("calibration", data_source, algorithm.replace("_interp", ""))
    recon_files = glob.glob(f"{recon_path}/*.npy")
    all_data = []
    for recon_file in recon_files:
        all_data.append(np.load(recon_file))
    target = np.asarray(all_data)

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
             p=p,
             target=target,
             p0=p0)

    return slope, intercept, r
