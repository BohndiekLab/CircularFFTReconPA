import glob
import numpy as np
from utils.constants import *


def load_all_recons(dataset, data_source, algorithm):
    base_path = get_recon_path(dataset, data_source, algorithm)
    all_files = glob.glob(base_path + "/*")
    all_data = np.zeros((len(all_files), 300, 300))
    for d_idx, data in enumerate(all_files):
        all_data[d_idx] = np.load(data)
    return all_data


def load_all(dataset, data_source):
    base_path = get_path(dataset, data_source)
    all_files = glob.glob(base_path + "/*")
    all_data = np.zeros((len(all_files), 300, 300))
    for d_idx, data in enumerate(all_files):
        all_data[d_idx] = np.load(data)
    return all_data
