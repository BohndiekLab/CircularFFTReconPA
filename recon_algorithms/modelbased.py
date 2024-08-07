import patato as pat
import numpy as np
from patato.io.ipasc.read_ipasc import IPASCInterface
from utils.constants import *
import glob
import time
import matplotlib.pyplot as plt

data_source = "testing"

def reconstruct(path, sound_speed=1500):
    pa_data = pat.PAData(IPASCInterface(PATH_IPASC_FILE))
    time_factor = 1
    detector_factor = 1
    preproc = pat.DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor,
                                          hilbert=False, lp_filter=None, hp_filter=None,
                                          irf=False)
    patato_recon = pat.ModelBasedReconstruction(field_of_view=(0.032, 0.032, 0.032),
                                                n_pixels=(300, 1, 300), regulariser="laplacian")

    ts = pa_data.get_time_series()
    time_series = np.load(path)
    ts.raw_data = np.reshape(time_series, (1, 1, 256, -1))
    new_t1, d1, _ = preproc.run(ts, pa_data)
    recon, _, _ = patato_recon.run(new_t1, pa_data, sound_speed,
                                   **d1)
    return (np.asarray(np.squeeze(recon.raw_data))).copy()


# times = []
# # reconstruct sim
# for file in glob.glob(get_raw_path(data_source, "sim") + "/*.npy"):
#     print(file)
#     save_file_path = file.replace("raw/sim", "recons/mb/sim")
#     t = time.time()
#     recon = reconstruct(file).T
#     times.append(time.time() - t)
#     np.save(save_file_path, recon)
# times = np.asarray(times)
# print(times)
# print(np.mean(times[1:]), np.std(times[1:]))
#
# times = []
# # reconstruct sim_raw
# for file in glob.glob(get_raw_path(data_source, "sim_raw") + "/*.npy"):
#     print(file)
#     save_file_path = file.replace("raw/sim_raw", "recons/mb/sim_raw")
#     t = time.time()
#     recon = reconstruct(file).T
#     times.append(time.time() - t)
#     np.save(save_file_path, recon)
# times = np.asarray(times)
# print(times)
# print(np.mean(times[1:]), np.std(times[1:]))

times = []
# reconstruct exp
for file in glob.glob(get_raw_path(data_source, "exp") + "/*.npy")[0:1]:
    print(file)
    save_file_path = file.replace("raw/exp", "recons/mb/exp")
    t = time.time()
    recon = reconstruct(file).T
    plt.figure(figsize=(2, 2))
    plt.imshow(recon)
    plt.axis("off")
    plt.savefig(save_file_path.replace(".npy", ".png"))
    plt.close()
    times.append(time.time() - t)
    np.save(save_file_path, recon)
times = np.asarray(times)
print(times)
print(np.mean(times[1:]), np.std(times[1:]))
