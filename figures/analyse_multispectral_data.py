from utils.constants import get_mouse_recon_path, PATH_BASE
import numpy as np
import matplotlib.pyplot as plt
from utils.linear_unmixing import linear_unmixing
from matplotlib_scalebar.scalebar import ScaleBar
from utils.histogram_colourbar import add_histogram_colorbar
import simpa as sp
from calibrate.calibrate import calibrate_to_p0
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert2
import string
import nrrd

SPACING = 0.10666666667

ALGORITHMS = ["MB", "ITTR_interp", "FFT"]
NAMES = ["Model-based",
         "Iterative TR",
         "Circular FFT"]
# There exist 8 scans in total (10 wavelengths each)
EXAMPLE_IMAGE = "Scan_23"
WAVELENGTH = np.asarray([700, 730, 750, 760, 770, 800, 820, 840, 850, 880])
WL_CUTOFF = 7
SO2_MIN = 50
SO2_MAX = 80

arterial_spectrum = np.asarray([sp.MOLECULE_LIBRARY.oxyhemoglobin(1.0).spectrum.get_value_for_wavelength(wl) for wl in WAVELENGTH])
arterial_spectrum = arterial_spectrum / arterial_spectrum[5]

MASK_LABELS = {
    1: "BACKGROUND",
    2: "BODY",
    3: "SPLEEN",
    4: "KIDNEY",
    5: "SPINE",
    6: "AORTA"
}
COLOURS = ["black", "lightgray", "pink", "purple", "orange", "red"]


fig, axes = plt.subplots(3, 3, figsize=(8, 7.5), layout="constrained")

axes = [axes[0, 0], axes[1, 0], axes[2, 0],
        axes[0, 1], axes[1, 1], axes[2, 1],
        axes[0, 2], axes[1, 2], axes[2, 2]]

selection = np.s_[25:275, 50:300]

lbl, _ = nrrd.read(f"{PATH_BASE}/data/mice/mask/{EXAMPLE_IMAGE}-labels.nrrd")
labels = np.ones((300, 300))
labels[6:-6, 6:-6] = lbl[:, :, 0]
labels = np.fliplr(labels)[selection]

for a_idx, algo in enumerate(ALGORITHMS):
    print(algo)
    data = np.load(get_mouse_recon_path(algo) + f"/{EXAMPLE_IMAGE}.npy")
    if algo != "BPH" and algo != "MB":
        data = np.asarray([data[i].T[selection] for i in range(len(data))])
    else:
        data = np.asarray([data[i][selection] for i in range(len(data))])
    sample_image = np.copy(data[5])

    # strategy 1: normalise the data to be between 0 and 1
    #data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # strategy 2: set negative pixels to a value close to 0
    # data[data<0] = 1e-5

    # strategy 3: compute the absolute (terrible)
    # data = np.abs(data)

    # strategy 4: compute the absolute of a 2D hilbert transform (terrible)
    # data = np.asarray([np.abs(hilbert2(i)) for i in data])

    # strategy 5: normalise per spectrum
    # data = (data - np.min(data, axis=0)[None, :, :]) / (np.max(data, axis=0)[None, :, :]- np.min(data, axis=0)[None, :, :])

    # strategy 6: ignore negative pixels
    #data[data <= 0] = np.nan

    # strategy 7: use the calibration
    # slope, intercept, r = calibrate_to_p0(algo.replace("_interp", ""), "exp")
    # data = intercept + slope * data

    # Normalise the spectra, such that the value at 800nm is 1

    data = data - np.mean(data)
    l1 = np.linalg.norm(np.reshape(data, (-1)), ord=1)/np.size(data)
    data = data + 4 * l1

    data[data <= 0] = np.nan

    data = data / data[5][None, :, :]
    print(np.shape(data))

    axes[a_idx].imshow(sample_image, cmap="viridis", vmin=np.percentile(sample_image, 1), vmax=np.percentile(sample_image, 99))
    axes[a_idx].axis("off")
    axes[a_idx].add_artist(ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="white", box_alpha=0))
    add_histogram_colorbar(axes[a_idx], sample_image, label="p$_0$", colormap="viridis",
                           vmin=np.percentile(sample_image, 1), vmax=np.percentile(sample_image, 99))

    for idx in [2, 3, 4, 5, 6]:
        axes[a_idx].plot([], [], color=COLOURS[idx-1], label=MASK_LABELS[idx])
        axes[a_idx].contour(labels==idx, colors=COLOURS[idx-1])

    # While showing the literature spectra for HBO2 would be cool to compare the reconstructed spectrum with,
    # it does not bring much of a benefit, as the reconstructed spectrum looks to be quite off for all the
    # reconstruction methods.

    # axes[a_idx].plot([], [], color="black", linestyle="dotted", label="HbO$_2$ (lit.)")
    # axes[a_idx + 3].plot(WAVELENGTH[:WL_CUTOFF], arterial_spectrum[:WL_CUTOFF], color="black", linestyle="dotted", label="HbO$_2$ (lit.)")
    for label in [3, 4, 5, 6]:
        axes[a_idx + 3].plot(WAVELENGTH[:WL_CUTOFF], np.nanmedian(data[:WL_CUTOFF, labels == label], axis=1), color=COLOURS[label - 1],
                             label=MASK_LABELS[label])
        axes[a_idx + 3].spines[["right", "top"]].set_visible(False)

    axes[a_idx + 3].set_ylim(0.78, 1.1)

    if a_idx == 0:
        axes[a_idx].text(40, 20, "Reconstruction", fontweight="bold", color="white")
        axes[a_idx].legend(ncol=1, loc="center right", labelspacing=0, fontsize=8.5,
                           borderpad=0.1, handlelength=0.8, handletextpad=0.4,
                           labelcolor="white", framealpha=0)
        axes[a_idx+3].legend(ncol=1, loc="lower right", labelspacing=0, fontsize=8.5,
                             borderpad=0.1, handlelength=0.8, handletextpad=0.4,
                             labelcolor="black", framealpha=0)
        axes[a_idx+3].text(720, 1.075, r"Avg. PA signal over $\lambda$", fontweight="bold", color="black")
        axes[a_idx + 6].text(40, 20, "Linearly unmixed sO$_2$", fontweight="bold", color="black")
    lu = np.squeeze(linear_unmixing(data[:WL_CUTOFF], WAVELENGTH[:WL_CUTOFF])) * 100
    lu[labels < 2] = np.nan
    axes[a_idx+6].imshow(lu, vmin=SO2_MIN, vmax=SO2_MAX, cmap="magma")
    axes[a_idx+6].axis("off")
    axes[a_idx+6].add_artist(
        ScaleBar(SPACING, "mm", length_fraction=0.25, location="lower left", color="black", box_alpha=0))
    add_histogram_colorbar(axes[a_idx+6], lu, label="sO$_2$", vmin=SO2_MIN, vmax=SO2_MAX,
                           min_label=str(SO2_MIN), max_label=str(SO2_MAX), colormap="magma")

    for idx in [2, 3, 4, 5, 6]:
        axes[a_idx+6].plot([], [], color=COLOURS[idx-1], label=f"{np.nanmean(lu[labels==idx]):.0f}$\pm$"
                                                               f"{np.nanstd(lu[labels==idx]):.0f}%")
        axes[a_idx+6].contour(labels == idx, colors=COLOURS[idx-1])

    axes[a_idx+6].legend(ncol=1, loc="center right", labelspacing=0, fontsize=8,
                         borderpad=0.1, handlelength=0.8, handletextpad=0.4,
                         labelcolor="black", framealpha=0)

for i in range(3):
    axes[i].text(-0.05, 0.5, NAMES[i], rotation=90, va='center', ha='right',
            transform=axes[i].transAxes, fontsize=12, fontweight="bold")


axes[0].text(0.03, 0.87, string.ascii_uppercase[0], transform=axes[0].transAxes,
        size=24, weight='bold', color="white")
axes[1].text(0.03, 0.87, string.ascii_uppercase[3], transform=axes[1].transAxes,
        size=24, weight='bold', color="white")
axes[2].text(0.03, 0.87, string.ascii_uppercase[6], transform=axes[2].transAxes,
        size=24, weight='bold', color="white")

axes[3].text(0.03, 0.87, string.ascii_uppercase[1], transform=axes[3].transAxes,
            size=24, weight='bold', color="black")
axes[4].text(0.03, 0.87, string.ascii_uppercase[4], transform=axes[4].transAxes,
            size=24, weight='bold', color="black")
axes[5].text(0.03, 0.87, string.ascii_uppercase[7], transform=axes[5].transAxes,
            size=24, weight='bold', color="black")

axes[6].text(0.03, 0.87, string.ascii_uppercase[2], transform=axes[6].transAxes,
            size=24, weight='bold', color="black")
axes[7].text(0.03, 0.87, string.ascii_uppercase[5], transform=axes[7].transAxes,
            size=24, weight='bold', color="black")
axes[8].text(0.03, 0.87, string.ascii_uppercase[8], transform=axes[8].transAxes,
            size=24, weight='bold', color="black")
# axes[9].text(0.03, 0.87, string.ascii_uppercase[5], transform=axes[9].transAxes,
#             size=24, weight='bold', color="black")
# axes[10].text(0.03, 0.87, string.ascii_uppercase[8], transform=axes[10].transAxes,
#             size=24, weight='bold', color="black")
# axes[11].text(0.03, 0.87, string.ascii_uppercase[11], transform=axes[11].transAxes,
#             size=24, weight='bold', color="black")


plt.savefig(f"mouse_data.pdf", dpi=300)
plt.show()
