import glob
import time
import patato as pat
from patato.recon.jax_model_based import JAXModelBasedReconstruction, ModelBasedPreProcessor
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils.constants import *
import tracemalloc

DATA_SOURCES = ["testing"]


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
    # Start tracing memory allocations
    tracemalloc.start()
    # reconstruct sim_raw
    for file in glob.glob(get_raw_path(data_source, "sim_raw") + "/*.npy")[0:1]:
        print(file)
        save_file_path = file.replace("raw/sim_raw", "recons/mb/sim_raw")
        t = time.time()
        recon = reconstruct(file).T
        times.append(time.time() - t)
    times = np.asarray(times)
    print(times)
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current Memory Usage: {current / (1024 * 1024):.2f} MB")
    print(f"Peak Memory Usage: {peak / (1024 * 1024):.2f} MB")
    tracemalloc.stop()

