import matplotlib.pyplot as plt
from utils.constants import get_recon_path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import json
import glob

algorithms = ["TR", "ITTR", "MB", "FFT", "BP", "BPH", "TR_interp", "ITTR_interp"]
datas = ["exp", "sim_raw"]

lines = {
    "P.5.11": np.asarray([[0, 299], [150, 150]]),
    "P.5.12": np.asarray([[106, 106], [0, 299]]),
    "P.5.14": np.asarray([[150, 150], [0, 299]]),
    "P.5.15": np.asarray([[0, 299], [0, 299]]),
    "P.5.19": np.asarray([[0, 299], [150, 150]]),
    "P.5.20": np.asarray([[215, 215], [0, 299]]),
    "P.5.23": np.asarray([[0, 299], [150, 150]]),
    "P.5.24": np.asarray([[0, 299], [150, 150]]),
    "P.5.29": np.asarray([[0, 299], [150, 150]]),
    "P.5.32": np.asarray([[125, 125], [0, 299]]),
    "P.5.3": np.asarray([[0, 299], [150, 150]]),
    "P.5.4.3": np.asarray([[0, 299], [150, 150]]),
    "P.5.6.2": np.asarray([[0, 299], [44, 268]]),
    "P.5.7": np.asarray([[0, 299], [150, 150]]),
    "P.5.8.2": np.asarray([[0, 113], [128, 297]]),
    "P.5.8.3": np.asarray([[0, 299], [150, 150]]),
}

results = {}

for data in datas:
    print(data)
    results[data] = dict()
    for key in algorithms:
        print(f"\t{key}")
        files = glob.glob(get_recon_path(data_set="testing", data=data, algorithm=key) + "/*.npy")
        intercept = np.load(f"../calibrate/cal_{key.replace('_interp', '')}_{data}.npz")["intercept"]
        slope = np.load(f"../calibrate/cal_{key.replace('_interp', '')}_{data}.npz")["slope"]
        fwhms = []
        for file in files:
            filename = file.split("\\")[-1].split("/")[-1].split("_")[0]
            image = np.load(file)
            image = intercept + slope * image
            line = lines[filename]
            num = 300
            x, y = np.linspace(line[0, 0], line[0, 1], num), np.linspace(line[1, 0], line[1, 1], num)
            line_values = map_coordinates(image, np.vstack((x, y)))

            gradient = np.abs(np.gradient(line_values))
            # emirically determined the acceptance threshold and Gaussian blur level
            # where to still count peaks.
            # This typically finds the edges and major peaks in the signal.
            # only very rarely will it count a peak twice.
            filtered_gradient = gaussian_filter1d(gradient, 3)
            peaks, _ = find_peaks(filtered_gradient, np.max(filtered_gradient)/2.5)
            peak_values = gradient[peaks]
            for peak, peak_value in zip(peaks, peak_values):
                left = peak
                left_value = peak_value
                while left_value >= peak_value / 2:
                    if left <= 0:
                        break
                    left = left - 1
                    left_value = gradient[left]
                right = peak
                right_value = peak_value
                while right_value >= peak_value / 2:
                    if right >= num-1:
                        break
                    right = right + 1
                    right_value = gradient[right]

                fwhms.append(float(right - left))

        results[data][key] = fwhms

with open("sharpness.json", "w+") as jsonfile:
    json.dump(results, jsonfile)