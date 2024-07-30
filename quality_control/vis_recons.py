import numpy as np
from utils.constants import *
import matplotlib.pyplot as plt
from scipy.signal import hilbert2

EXAMPLE_PHANTOM = "P.5.10.2_700.npy"

p0_orig = np.load(f"{get_raw_path('calibration', 'p0')}/{EXAMPLE_PHANTOM}")
tr_orig = np.load(f"{get_recon_path('calibration', 'sim_raw', 'tr')}/{EXAMPLE_PHANTOM}")
ittr_orig = np.load(f"{get_recon_path('calibration', 'sim_raw', 'ittr')}/{EXAMPLE_PHANTOM}")
fft_orig = np.load(f"{get_recon_path('calibration', 'sim_raw', 'fft')}/{EXAMPLE_PHANTOM}")
bp_orig = np.load(f"{get_recon_path('calibration', 'sim_raw', 'bp')}/{EXAMPLE_PHANTOM}")
mb_orig = np.load(f"{get_recon_path('calibration', 'sim_raw', 'mb')}/{EXAMPLE_PHANTOM}")

p0 = p0_orig.copy()
tr = tr_orig.copy()
ittr = ittr_orig.copy()
fft = fft_orig.copy()
bp = bp_orig.copy()
mb = mb_orig.copy()

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

segmentation_mask = np.zeros_like(p0).astype(bool)
segmentation_mask[:75, :] = p0[:75, :] < 20
segmentation_mask[225:, :] = p0[225:, :] < 20
segmentation_mask[:, :75] = p0[:, :75] < 20
segmentation_mask[:, 225:] = p0[:, 225:] < 20

p0[segmentation_mask] = np.nan
tr[segmentation_mask] = np.nan
ittr[segmentation_mask] = np.nan
fft[segmentation_mask] = np.nan
bp[segmentation_mask] = np.nan
mb[segmentation_mask] = np.nan

ax1.imshow(p0)
ax2.imshow(tr)
ax3.imshow(ittr)
ax4.imshow(bp)
ax5.imshow(mb)
ax6.imshow(fft)


def decorate(axis, title):
    axis.axis("off")


decorate(ax1, "Initial Pressure")
decorate(ax2, "Time Reversal")
decorate(ax3, "Iterative TR")
decorate(ax4, "Backprojection")
decorate(ax5, "Model-based")
decorate(ax6, "FFT-based")

plt.show()