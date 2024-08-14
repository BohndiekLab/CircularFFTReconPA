import numpy as np
from utils.constants import *
import matplotlib.pyplot as plt

EXAMPLE_PHANTOM = "P.5.11_700.npy"

p0_orig = np.load(f"{get_p0_path('testing')}/{EXAMPLE_PHANTOM}")
tr_orig = np.load(f"{get_recon_path('testing', 'sim_raw', 'tr')}/{EXAMPLE_PHANTOM}")
ittr_orig = np.load(f"{get_recon_path('testing', 'sim_raw', 'ittr')}/{EXAMPLE_PHANTOM}")
fft_orig = np.load(f"{get_recon_path('testing', 'sim_raw', 'fft')}/{EXAMPLE_PHANTOM}")
bp_orig = np.load(f"{get_recon_path('testing', 'sim_raw', 'bp')}/{EXAMPLE_PHANTOM}")
mb_orig = np.load(f"{get_recon_path('testing', 'sim_raw', 'mb')}/{EXAMPLE_PHANTOM}")

p0 = p0_orig.copy()
tr = tr_orig.copy()
ittr = ittr_orig.copy()
fft = fft_orig.copy()
bp = bp_orig.copy()
mb = mb_orig.copy()

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.set_title("p0")
ax1.imshow(p0)
ax2.set_title("TR")
ax2.imshow(tr)
ax3.set_title("ITTR")
ax3.imshow(ittr)
ax4.set_title("BP")
ax4.imshow(bp)
ax5.set_title("MB")
ax5.imshow(mb)
ax6.set_title("FFT")
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