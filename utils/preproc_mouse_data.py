import numpy as np
import matplotlib.pyplot as plt
from patato.io.msot_data import PAData


SCANS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

for scan in SCANS:
    path = rf"H:\albino_mice\Scan_{scan}.hdf5"

    data = PAData.from_hdf5(path)
    print(data.get_wavelengths())
    exit()
    raw_data = data.get_time_series().raw_data
    print(np.shape(raw_data))
    mouse_data = raw_data[5, :, :, :]

    np.save(rf"H:\albino_mice\raw_numpy\Scan_{scan}.npy", mouse_data)