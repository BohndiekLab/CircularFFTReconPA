import numpy as np
from utils.constants import get_recon_path, get_mouse_recon_path
import os, glob
import matplotlib.pyplot as plt

dss = ["calibration", "testing"]
dats = ["exp", "sim", "sim_raw"]
algorithms = ["tr", "ittr", "mb", "fft", "bp", "bph", "tr_interp", "ittr_interp"]

# Phantom Data
# print("Quality checking phantom reconstructions.")
# for ds in dss:
#     for dat in dats:
#         for algorithm in algorithms:
#             folder = get_recon_path(data_set=ds, data=dat, algorithm=algorithm)
#             if not os.path.exists(folder):
#                 print(f"The following data does not exist: {ds}, {dat}, {algorithm}")
#                 continue
#
#             files = glob.glob(f"{folder}/*.npy")
#             if ds == "calibration":
#                 if len(files) != 75:
#                     print(f"\t{ds}, {dat}, {algorithm} had {len(files)} files instead of 75")
#             if ds == "testing":
#                 if len(files) != 79:
#                     print(f"\t{ds}, {dat}, {algorithm} had {len(files)} files instead of 79")
#
#             for file in files:
#                 data = np.load(file)
#                 # plt.figure(figsize=(2, 2))
#                 # plt.imshow(data)
#                 # plt.axis("off")
#                 # plt.savefig(file.replace(".npy", ".png"))
#                 # plt.close()
#                 if np.shape(data) != (300, 300):
#                     print(f"\t\t{ds}, {dat}, {algorithm}: Found a file that was not the correct dimensions (300, 300). "
#                           f"Instead was: {np.shape(data)}")
# print("Found no further inconsistencies.")

# Mouse Data
print("Quality checking mouse reconstructions.")
for algorithm in algorithms:
    folder = get_mouse_recon_path(algorithm=algorithm)
    if not os.path.exists(folder):
        print(f"The following mouse data does not exist: {algorithm}")
        continue

    files = glob.glob(f"{folder}/*.npy")

    if len(files) != 8:
        print(f"\t{algorithm} had {len(files)} files instead of 8")

    for file in files:
        data = np.load(file)
        plt.figure(figsize=(2, 2))
        plt.imshow(data[0])
        plt.axis("off")
        plt.savefig(file.replace(".npy", ".png"))
        plt.close()
        if np.shape(data) != (10, 300, 300):
            print(f"\t\t{algorithm}: Found a file that was not the correct dimensions (10, 300, 300). "
                  f"Instead was: {np.shape(data)}")

print("Found no further inconsistencies.")