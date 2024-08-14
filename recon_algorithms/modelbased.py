import glob
import time
import patato as pat
from patato.recon.jax_model_based import JAXModelBasedReconstruction, ModelBasedPreProcessor
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils.constants import *

DATA_SOURCES = ["calibration", "testing"]

def reconstruct(path, sound_speed=1488):
    f = h5py.File(PATH_IPASC_FILE)
    geom = np.array([x["detector_position"][:] for _, x in f['meta_data_device']["detectors"].items()])

    dataset = np.load(path)[None, None]
    wavelengths = [700]
    fs = 4e7

    ts = pat.PATimeSeries.from_numpy(dataset, wavelengths, fs, sound_speed)

    pre = ModelBasedPreProcessor()
    ts_prime, settings, _ = pre.run(ts, None)

    model_params = {"model_geometry": geom,
                    "model_fs": 4e7,
                    "model_c": sound_speed,
                    "model_irf": None,
                    "model_nt": 2030,
                    "model_constraint": "none"
                    }

    m = JAXModelBasedReconstruction(field_of_view=(0.032, 0., 0.032), n_pixels=(300, 1, 300), **model_params)
    recon, _, _ = m.run(ts, None)

    return (np.asarray(np.squeeze(recon.raw_data))).copy()

for data_source in DATA_SOURCES:
    times = []
    # reconstruct sim_raw
    for file in glob.glob(get_raw_path(data_source, "sim_raw") + "/*.npy"):
        print(file)
        save_file_path = file.replace("raw/sim_raw", "recons/mb/sim_raw")
        t = time.time()
        recon = reconstruct(file).T
        times.append(time.time() - t)
        np.save(save_file_path, recon)
    times = np.asarray(times)
    print(times)
    print(np.mean(times[1:]), np.std(times[1:]))

    times = []
    # reconstruct exp
    for file in glob.glob(get_raw_path(data_source, "exp") + "/*.npy"):
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
