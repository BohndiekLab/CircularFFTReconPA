from utils.constants import get_mouse_recon_path
import matplotlib.pyplot as plt
import numpy as np
import glob

for algo in ["bp", "mb"]:
    path = get_mouse_recon_path(algo)
    scans = glob.glob(f"{path}/*.npy")
    for scan in scans:
        data = np.load(scan)
        data = data.swapaxes(0, 2)
        data = data.swapaxes(1, 2)
        print(np.shape(data))
        np.save(scan, data)
