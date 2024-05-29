from utils.constants import get_recon_path, PATH_BASE
import numpy as np
import matplotlib.pyplot as plt
from utils.linear_unmixing import linear_unmixing
from matplotlib_scalebar.scalebar import ScaleBar
from utils.histogram_colourbar import add_histogram_colorbar
import string
import nrrd

SPACING = 0.10666666667

ALGORITHMS = ["BP", "MB", "BP", "BP"]
NAMES = ["Backprojection",
         "Model-based",
         "Iterative TR",
         "Circular FFT"]
# There exist 8 scans in total (10 wavelengths each)
EXAMPLE_IMAGE = "Scan_23"

MASK_LABELS = {
    1: "BACKGROUND",
    2: "BODY",
    3: "SPLEEN",
    4: "KIDNEY",
    5: "SPINE",
    6: "ARTERY"
}
COLOURS = ["black", "lightgray", "pink", "purple", "orange", "red"]


fig, axes = plt.subplots(4, 2, figsize=(5.33, 10.66), layout="constrained")

axes = [axes[0, 0], axes[1, 0], axes[2, 0], axes[3, 0],
        axes[0, 1], axes[1, 1], axes[2, 1], axes[3, 1]]

selection = np.s_[25:275, 50:300]

lbl, _ = nrrd.read(f"{PATH_BASE}/data/mice/mask/{EXAMPLE_IMAGE}-labels.nrrd")
labels = np.ones((300, 300))
labels[6:-6, 6:-6] = lbl[:, :, 0]
labels = np.fliplr(labels)[selection]

for a_idx, algo in enumerate(ALGORITHMS):
    print(algo)
    data = np.load(get_recon_path("mice", "exp", algo) + f"/{EXAMPLE_IMAGE}.npy").T
    print(np.shape(data))

    sample_image = data[4][selection]
    axes[a_idx].imshow(sample_image, cmap="magma")
    axes[a_idx].axis("off")
    axes[a_idx].text(40, 20, NAMES[a_idx], fontweight="bold", color="white")
    axes[a_idx].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    add_histogram_colorbar(axes[a_idx], sample_image, label="p$_0$", colormap="magma")

    for idx in [2, 3, 4, 5, 6]:
        axes[a_idx].plot([], [], color=COLOURS[idx-1], label=MASK_LABELS[idx])
        axes[a_idx].contour(labels==idx, colors=COLOURS[idx-1])

    if a_idx == 0:
        axes[a_idx].legend(ncol=1, loc="center right", labelspacing=0, fontsize=8.5,
                  borderpad=0.1, handlelength=0.8, handletextpad=0.4,
                  labelcolor="white", framealpha=0)
    lu = np.squeeze(linear_unmixing(data, [700, 730, 750, 760, 770, 800, 820, 840, 850, 880])) * 100
    lu = lu[selection]
    lu[labels < 2] = np.nan
    axes[a_idx+4].imshow(lu, vmin=0, vmax=100)
    axes[a_idx+4].axis("off")
    axes[a_idx+4].text(40, 20, NAMES[a_idx] + " (sO$_2$)", fontweight="bold", color="black")
    axes[a_idx+4].add_artist(
        ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="black", box_alpha=0))
    add_histogram_colorbar(axes[a_idx+4], lu, label="sO$_2$", vmin=0, vmax=100,
                           min_label="0", max_label="100")

    for idx in [2, 3, 4, 5, 6]:
        axes[a_idx+4].plot([], [], color=COLOURS[idx-1], label=f"{np.nanmean(lu[labels==idx]):.0f}$\pm$"
                                                               f"{np.nanstd(lu[labels==idx]):.0f}%")
        axes[a_idx+4].contour(labels == idx, colors=COLOURS[idx-1])

    axes[a_idx+4].legend(ncol=1, loc="center right", labelspacing=0, fontsize=8,
              borderpad=0.1, handlelength=0.8, handletextpad=0.4,
              labelcolor="black", framealpha=0)


axes[0].text(0.03, 0.87, string.ascii_uppercase[0], transform=axes[0].transAxes,
        size=24, weight='bold', color="white")
axes[1].text(0.03, 0.87, string.ascii_uppercase[2], transform=axes[1].transAxes,
        size=24, weight='bold', color="white")
axes[2].text(0.03, 0.87, string.ascii_uppercase[4], transform=axes[2].transAxes,
        size=24, weight='bold', color="white")
axes[3].text(0.03, 0.87, string.ascii_uppercase[6], transform=axes[3].transAxes,
        size=24, weight='bold', color="white")
axes[4].text(0.03, 0.87, string.ascii_uppercase[1], transform=axes[4].transAxes,
            size=24, weight='bold', color="black")
axes[5].text(0.03, 0.87, string.ascii_uppercase[3], transform=axes[5].transAxes,
            size=24, weight='bold', color="black")
axes[6].text(0.03, 0.87, string.ascii_uppercase[5], transform=axes[6].transAxes,
            size=24, weight='bold', color="black")
axes[7].text(0.03, 0.87, string.ascii_uppercase[7], transform=axes[7].transAxes,
            size=24, weight='bold', color="black")


plt.savefig(f"mouse_data.pdf", dpi=300)
plt.show()
