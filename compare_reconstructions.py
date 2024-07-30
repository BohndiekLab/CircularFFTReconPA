from calibrate.calibrate import calibrate_to_p0
from utils.constants import get_p0_path, get_recon_path
import numpy as np
import matplotlib.pyplot as plt
from quality_control.measures import StructuralSimilarityIndex, JensenShannonDivergence
from quality_control.create_metric_window import apply_window_function
from matplotlib_scalebar.scalebar import ScaleBar
from utils.histogram_colourbar import add_histogram_colorbar
import string
import glob

SPACING = 0.10666666667

ALGORITHMS = ["BP", "BPH", "MB", "TR", "ITTR", "FFT"]
NAMES = ["Delay And Sum",
         "Filtered Backprojection",
         "Model-based",
         "Time reversal",
         "Iterative time reversal",
         "Circular FFT"]
data_source = "sim_raw"
EXAMPLE_IMAGE_IDX = 50

fig, axes = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")

axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1]]

p0_path = get_p0_path("calibration")
p0_files = glob.glob(f"{p0_path}/*.npy")
gt = []
for p0_file in p0_files:
    gt.append(np.load(p0_file))
gt = np.asarray(gt)
print(np.min(gt), np.max(gt))
gt_img = gt[EXAMPLE_IMAGE_IDX].copy()

axes[0].imshow(gt_img.T, vmin=np.nanmin(gt_img), vmax=np.nanmax(gt_img))
axes[0].axis("off")
axes[0].text(40, 20, "Simulated p$_0$", fontweight="bold", color="white")
axes[0].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
add_histogram_colorbar(axes[0], gt_img.T, label="p$_0$")

for a_idx, algo in enumerate(ALGORITHMS):
    print(algo)
    slope, intercept, r = calibrate_to_p0(algo, data_source)
    recon_path = get_recon_path("calibration", data_source, algo)
    recon_files = glob.glob(f"{recon_path}/*.npy")
    all_data = []
    for recon_file in recon_files:
        all_data.append(np.load(recon_file))
    all_data = np.asarray(all_data)
    print(np.min(all_data), np.max(all_data))
    images = intercept + slope * all_data

    sample_image = images[EXAMPLE_IMAGE_IDX].copy()
    axes[a_idx + 1].imshow(sample_image.T, vmin=np.nanmin(gt_img), vmax=np.nanmax(gt_img))
    axes[a_idx + 1].axis("off")
    axes[a_idx + 1].text(40, 20, NAMES[a_idx], fontweight="bold", color="white")
    axes[a_idx + 1].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    add_histogram_colorbar(axes[a_idx + 1], sample_image.T, vmin=np.min(gt_img),
                           vmax=np.max(gt_img), label="p$_0$")

axes[0].text(0.03, 0.87, string.ascii_uppercase[0], transform=axes[0].transAxes,
        size=24, weight='bold', color="white")
axes[1].text(0.03, 0.87, string.ascii_uppercase[2], transform=axes[1].transAxes,
        size=24, weight='bold', color="white")
axes[2].text(0.03, 0.87, string.ascii_uppercase[4], transform=axes[2].transAxes,
        size=24, weight='bold', color="white")
axes[3].text(0.03, 0.87, string.ascii_uppercase[1], transform=axes[3].transAxes,
        size=24, weight='bold', color="white")
axes[4].text(0.03, 0.87, string.ascii_uppercase[3], transform=axes[4].transAxes,
        size=24, weight='bold', color="white")
axes[5].text(0.03, 0.87, string.ascii_uppercase[5], transform=axes[5].transAxes,
        size=24, weight='bold', color="white")

plt.savefig(f"recon_examples_{data_source}.pdf", dpi=300)
plt.show()
