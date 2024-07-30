from utils.constants import get_mouse_recon_path
import numpy as np

path = get_mouse_recon_path("fft_single")

scans = [3, 5, 7, 9, 15, 17, 21, 23]

for scan in scans:
    np_datas = []
    for idx in range(10):
        data = np.load(f"{path}/Scan_{scan}_{idx}.npy")
        np_datas.append(data)
    print(np.shape(np_datas))
    np.save(f"{path.replace('fft_single', 'fft')}/Scan_{scan}.npy", np.asarray(np_datas))