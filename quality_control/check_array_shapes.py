import numpy as np
from utils.constants import *

EXAMPLE_PHANTOM = "P.5.10.2_700.npy"

p0 = np.load(f"{PATH_CAL_P0}/{EXAMPLE_PHANTOM}")
tr = np.load(f"{PATH_CAL_REC_TR_SIM}/{EXAMPLE_PHANTOM}")
ittr = np.load(f"{PATH_CAL_REC_ITTR_SIM}/{EXAMPLE_PHANTOM}")
fft = np.load(f"{PATH_CAL_REC_FFT_SIM}/{EXAMPLE_PHANTOM}")
bp = np.load(f"{PATH_CAL_REC_BP_SIM}/{EXAMPLE_PHANTOM}")
mb = np.load(f"{PATH_CAL_REC_MB_SIM}/{EXAMPLE_PHANTOM}")

print(np.shape(p0))
print(np.shape(tr))
print(np.shape(ittr))
print(np.shape(fft))
print(np.shape(bp))
print(np.shape(mb))
