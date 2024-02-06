import numpy as np
from utils.constants import *
import matplotlib.pyplot as plt
from scipy.signal import hilbert2

EXAMPLE_PHANTOM = "P.5.10.2_700.npy"

p0 = np.load(f"{PATH_CAL_P0}/{EXAMPLE_PHANTOM}")
tr = np.load(f"{PATH_CAL_REC_TR_SIM}/{EXAMPLE_PHANTOM}")
ittr = np.load(f"{PATH_CAL_REC_ITTR_SIM}/{EXAMPLE_PHANTOM}")
fft = np.load(f"{PATH_CAL_REC_FFT_SIM}/{EXAMPLE_PHANTOM}")
bp = np.load(f"{PATH_CAL_REC_BP_SIM}/{EXAMPLE_PHANTOM}")
mb = np.load(f"{PATH_CAL_REC_MB_SIM}/{EXAMPLE_PHANTOM}")

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.imshow(p0)
ax2.imshow(tr)
ax3.imshow(ittr)
ax4.imshow(bp)
ax5.imshow(mb)
ax6.imshow(fft)

plt.show()