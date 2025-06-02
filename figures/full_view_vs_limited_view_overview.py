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
ALGORITHMS = ["mb", "ittr", "fft"]
ALGO_TEXT = ["Model-based", "Iterative time reversal", "Circular FFT"]

fig, axes = plt.subplots(3, 2, layout="constrained", figsize=(5.5, 8.25))

for idx, (ALGORITHM, ax) in enumerate(zip(ALGORITHMS, axes)):
    print(idx, ALGORITHM)
    example_limited = np.load(get_recon_path(data_set="testing", data="sim_raw", algorithm=ALGORITHM) + "/P.5.24_800.npy").T
    example_full = np.load(get_full_view_path(ALGORITHM) + "/P.5.24_800.npy").T
    print("\t", np.min(example_full))
    print("\t", np.min(example_limited))

    example_limited = example_limited - np.min(example_limited)
    example_full = example_full - np.min(example_full)

    # images and colourbars
    ax[0].imshow(example_full, cmap="viridis")
    ax[0].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    add_histogram_colorbar(ax[0], example_full, label="PAI", colormap="viridis", fontsize=10)
    ax[0].axis("off")

    ax[1].imshow(example_limited, cmap="viridis")
    ax[1].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    add_histogram_colorbar(ax[1], example_limited, label="PAI", colormap="viridis", fontsize=10)
    ax[1].axis("off")

    ax[0].text(0.03, 0.87, string.ascii_uppercase[2*idx], transform=ax[0].transAxes,
            size=24, weight='bold', color="white")
    ax[1].text(0.03, 0.87, string.ascii_uppercase[(2*idx) + 1], transform=ax[1].transAxes,
            size=24, weight='bold', color="white")

axes[0][0].set_title("Full-view $(360^\circ)$", fontsize=12, fontweight="bold")
axes[0][1].set_title("Limited view $(270^\circ)$", fontsize=12, fontweight="bold")

for i in range(3):
    axes[i][0].text(-0.02, 0.5, ALGO_TEXT[i], rotation=90, va='center', ha='right',
                    transform=axes[i][0].transAxes, fontsize=12, fontweight="bold")

plt.savefig(f"full_view_vs_limited_view.pdf", dpi=300)

plt.show()