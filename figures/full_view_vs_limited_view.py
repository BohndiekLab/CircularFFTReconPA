from utils.constants import get_full_view_path, get_p0_path, get_recon_path
from utils.segmentation import get_coupling_medium_segmentation
from utils.histogram_colourbar import add_histogram_colorbar
from matplotlib_scalebar.scalebar import ScaleBar
from calibrate.calibrate import calibrate_to_p0
import matplotlib.pyplot as plt
import numpy as np
import string
import json
import glob
import os
from measures.measures import StructuralSimilarityIndex, MedianAbsoluteError, \
    JensenShannonDivergence, HaarPSI
from scipy.stats import linregress

SPACING = 0.10666666667
ALGORITHM = "fft"

if os.path.exists(f"res_full_{ALGORITHM}.json") and os.path.exists(f"res_limited_{ALGORITHM}.json"):

    with open(f"res_full_{ALGORITHM}.json", "r+") as jsonfile:
        res_full = json.load(jsonfile)

    with open(f"res_limited_{ALGORITHM}.json", "r+") as jsonfile:
        res_limited = json.load(jsonfile)

else:

    res_full = dict()
    res_full["R"] = []
    res_full["MAE"] = []
    res_full["SSIM"] = []
    res_full["JSD"] = []
    res_full["HaarPSI"] = []

    res_limited = dict()
    res_limited["R"] = []
    res_limited["MAE"] = []
    res_limited["SSIM"] = []
    res_limited["JSD"] = []
    res_limited["HaarPSI"] = []

    files = glob.glob(get_full_view_path(ALGORITHM) + "/*.npy")

    slope, intercept, r = calibrate_to_p0(ALGORITHM, "sim_raw")
    print(slope)
    print(intercept)
    print(r)

    for file in files:
        filename = file.split("\\")[-1].split("/")[-1]
        print(filename)
        full_view = np.load(file)
        if os.path.exists(get_recon_path("testing", "sim_raw", ALGORITHM) + f"/{filename}"):
            limited_view = np.load(get_recon_path("testing", "sim_raw", ALGORITHM) + f"/{filename}")
        else:
            limited_view = np.load(get_recon_path("calibration", "sim_raw", ALGORITHM) + f"/{filename}")

        if os.path.exists(get_p0_path("testing") + f"/{filename}"):
            p0 = np.load(get_p0_path("testing") + f"/{filename}")
        else:
            p0 = np.load(get_p0_path("calibration") + f"/{filename}")

        full_view = full_view * slope + intercept
        limited_view = limited_view * slope + intercept

        water_segmentation = get_coupling_medium_segmentation(p0)
        # plt.subplot(1, 4, 1)
        # plt.imshow(water_segmentation)
        # plt.subplot(1, 4, 2)
        # plt.imshow(p0)
        # plt.subplot(1, 4, 3)
        # plt.imshow(full_view)
        # plt.subplot(1, 4, 4)
        # plt.imshow(limited_view)
        # plt.show()
        # plt.close()
        # exit()

        print("\tR")
        p0[~water_segmentation] = np.nan
        full_view[~water_segmentation] = np.nan
        limited_view[~water_segmentation] = np.nan

        _, _, r, _, _ = linregress(full_view.reshape((-1,))[~np.isnan(full_view.reshape((-1,)))],
                                   p0.reshape((-1,))[~np.isnan(p0.reshape((-1,)))])
        res_full["R"].append(r)
        _, _, r, _, _ = linregress(limited_view.reshape((-1,))[~np.isnan(limited_view.reshape((-1,)))],
                                   p0.reshape((-1,))[~np.isnan(p0.reshape((-1,)))])
        res_limited["R"].append(r)

        print("\tMAE")
        p0[~water_segmentation] = np.nan
        full_view[~water_segmentation] = np.nan
        limited_view[~water_segmentation] = np.nan
        res_full["MAE"].append(MedianAbsoluteError(full_view, p0))
        res_limited["MAE"].append(MedianAbsoluteError(limited_view, p0))

        print("\tSSIM")
        p0[~water_segmentation] = 0
        full_view[~water_segmentation] = 0
        limited_view[~water_segmentation] = 0
        res_full["SSIM"].append(StructuralSimilarityIndex(full_view, p0))
        res_limited["SSIM"].append(StructuralSimilarityIndex(limited_view, p0))

        print("\tJSD")
        p0[~water_segmentation] = np.nan
        full_view[~water_segmentation] = np.nan
        limited_view[~water_segmentation] = np.nan
        res_full["JSD"].append(JensenShannonDivergence(full_view.reshape((1, -1)), p0.reshape((1, -1))))
        res_limited["JSD"].append(JensenShannonDivergence(limited_view.reshape((1, -1)), p0.reshape((1, -1))))

        print("\tHaarPSI")
        p0[~water_segmentation] = 0
        full_view[~water_segmentation] = 0
        limited_view[~water_segmentation] = 0
        res_full["HaarPSI"].append(HaarPSI(full_view, p0))
        res_limited["HaarPSI"].append(HaarPSI(limited_view, p0))

    with open(f"res_full_{ALGORITHM}.json", "w+") as jsonfile:
        json.dump(res_full, jsonfile)

    with open(f"res_limited_{ALGORITHM}.json", "w+") as jsonfile:
        json.dump(res_limited, jsonfile)

example_p0 = np.load(get_p0_path("testing") + "/P.5.24_800.npy").T
example_limited = np.load(get_recon_path(data_set="testing", data="sim_raw", algorithm=ALGORITHM) + "/P.5.24_800.npy").T
example_full = np.load(get_full_view_path(ALGORITHM) + "/P.5.24_800.npy").T

example_limited = example_limited - np.min(example_limited)
example_full = example_full - np.min(example_full)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, layout="constrained", figsize=(5.5, 5.5))

# images and colourbars
ax1.imshow(example_p0, cmap="viridis")
ax1.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
add_histogram_colorbar(ax1, example_p0, label="p$_0$", colormap="viridis", fontsize=10)
ax1.axis("off")
ax3.imshow(example_full, cmap="viridis")
ax3.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
add_histogram_colorbar(ax3, example_full, label="PAI", colormap="viridis", fontsize=10)
ax3.axis("off")
ax4.imshow(example_limited, cmap="viridis")
ax4.add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
add_histogram_colorbar(ax4, example_limited, label="PAI", colormap="viridis", fontsize=10)
ax4.axis("off")

# Statistics and measures
ax2.axis("off")

ax2.text(0.25, 0.75, f"Metric", fontweight="bold", ha="right", size=11)
ax2.text(0.3, 0.75, f"360$^\circ$", fontweight="bold", size=11)
ax2.text(0.5, 0.75, f"270$^\circ$", fontweight="bold", size=11)
ax2.text(0.725, 0.75, f"Change", fontweight="bold", size=11)

for idx, data_key in enumerate(res_full.keys()):
    ax2.text(0.25, 0.65-0.1*idx, data_key, fontweight="bold", ha="right", size=11)
    full_data = np.mean(res_full[data_key])
    limited_data = np.mean(res_limited[data_key])
    ax2.text(0.3, 0.65 - 0.1 * idx, f"{full_data:.2f}", size=11)
    ax2.text(0.5, 0.65 - 0.1 * idx, f"{limited_data:.2f}", size=11)
    ax2.text(0.725, 0.65 - 0.1 * idx, f"{((limited_data-full_data)/full_data) * 100:.1f}%", size=11)


ax1.text(0.03, 0.9, string.ascii_uppercase[0], transform=ax1.transAxes,
        size=24, weight='bold', color="white")
ax2.text(0.03, 0.9, string.ascii_uppercase[1], transform=ax2.transAxes,
        size=24, weight='bold', color="black")
ax3.text(0.03, 0.9, string.ascii_uppercase[2], transform=ax3.transAxes,
        size=24, weight='bold', color="white")
ax4.text(0.03, 0.9, string.ascii_uppercase[3], transform=ax4.transAxes,
        size=24, weight='bold', color="white")

plt.savefig(f"full_view_vs_limited_view_{ALGORITHM}.pdf", dpi=300)

plt.show()