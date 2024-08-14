import numpy as np
import os

INPUT = r"F:\fft_recon_project\fft_mouse_data"
OUTPUT = r"F:\fft_recon_project\data\mice\recons\fft"
DATA = [3, 5, 7, 9, 15, 17, 21, 23]
FRAME_IDS = np.asarray(list(range(10)))
for scan_id in DATA:
    print(f"Scan_{scan_id}")
    frames = []
    for frame in FRAME_IDS:
        recon = np.squeeze(np.load(f"{INPUT}/Scan_{scan_id}_{frame}.npy"))
        frames.append(recon)
    np.save(f"{OUTPUT}/Scan_{scan_id}.npy",
            np.asarray(frames))


