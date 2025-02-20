from calibrate.calibrate import calibrate_to_p0
from utils.constants import get_p0_path, get_recon_path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from utils.histogram_colourbar import add_histogram_colorbar
import string
import glob

SPACING = 0.10666666667

ALGORITHMS = ["MB", "ITTR_interp", "FFT"]
NAMES = ["Model-based",
         "Iterative time reversal",
         "Circular FFT"]
data_sources = ["sim", "exp"]
EXAMPLE_IMAGE_IDX = "19"

fig, axes = plt.subplots(3, 2, figsize=(5.5, 8), layout="constrained")

all_axes = [[axes[0, 0], axes[1, 0], axes[2, 0]], [axes[0, 1], axes[1, 1], axes[2, 1]]]

p0_path = get_p0_path("testing")
p0_file = f"{p0_path}/P.5.{EXAMPLE_IMAGE_IDX}_800.npy"
gt = np.load(p0_file)
gt_img = gt.copy()

# axes[0].imshow(gt_img.T, vmin=np.nanmin(gt_img), vmax=np.nanmax(gt_img))
# axes[0].axis("off")
# axes[0].text(40, 20, "Simulated p$_0$", fontweight="bold", color="white")
# axes[0].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
# add_histogram_colorbar(axes[0], gt_img.T, label="p$_0$")

for axes, data_source in zip(all_axes, data_sources):
    for a_idx, algo in enumerate(ALGORITHMS):
        print(algo)
        slope, intercept, r = calibrate_to_p0(algo, data_source)
        recon_path = get_recon_path("testing", data_source, algo)
        recon_file = f"{recon_path}/P.5.{EXAMPLE_IMAGE_IDX}_800.npy"
        all_data = np.load(recon_file)
        images = intercept + slope * all_data
        sample_image = images.copy()

        axes[a_idx].imshow(sample_image.T, vmin=np.nanpercentile(gt_img, 1), vmax=np.nanpercentile(gt_img, 99))
        axes[a_idx].axis("off")
        #axes[a_idx].text(40, 20, NAMES[a_idx], fontweight="bold", color="white")
        axes[a_idx].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
        add_histogram_colorbar(axes[a_idx], sample_image.T, vmin=np.percentile(gt_img, 1),
                               vmax=np.percentile(gt_img, 99), label="p$_0$")
all_axes[0][0].set_title("Simulation", fontsize=12, fontweight="bold")
all_axes[1][0].set_title("Experiment", fontsize=12, fontweight="bold")

for i in range(3):
    all_axes[0][i].text(-0.02, 0.5, NAMES[i], rotation=90, va='center', ha='right',
                        transform=all_axes[0][i].transAxes, fontsize=12, fontweight="bold")

all_axes[0][0].text(0.03, 0.87, string.ascii_uppercase[0], transform=all_axes[0][0].transAxes,
                    size=24, weight='bold', color="white")
all_axes[1][0].text(0.03, 0.87, string.ascii_uppercase[1], transform=all_axes[1][0].transAxes,
                    size=24, weight='bold', color="white")
all_axes[0][1].text(0.03, 0.87, string.ascii_uppercase[2], transform=all_axes[0][1].transAxes,
                    size=24, weight='bold', color="white")
all_axes[1][1].text(0.03, 0.87, string.ascii_uppercase[3], transform=all_axes[1][1].transAxes,
                    size=24, weight='bold', color="white")
all_axes[0][2].text(0.03, 0.87, string.ascii_uppercase[4], transform=all_axes[0][2].transAxes,
                    size=24, weight='bold', color="white")
all_axes[1][2].text(0.03, 0.87, string.ascii_uppercase[5], transform=all_axes[1][2].transAxes,
                    size=24, weight='bold', color="white")

plt.savefig(f"recon_examples_P.5.{EXAMPLE_IMAGE_IDX}.pdf", dpi=300)
plt.show()
