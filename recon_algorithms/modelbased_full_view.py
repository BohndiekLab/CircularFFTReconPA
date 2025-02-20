import glob
import time
import patato as pat
from patato.recon.jax_model_based import JAXModelBasedReconstruction, ModelBasedPreProcessor
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from utils.constants import PATH_IPASC_FILE, get_full_view_path


def reconstruct(path, sound_speed=1488):
    geometry_path = Path(path).parent.parent.joinpath("detector_positions.txt")
    geom = np.loadtxt(geometry_path)
    geom = geom[::-1]
    # geom = np.array([x["detector_position"][:] for _, x in f['meta_data_device']["detectors"].items()])
    # print(geom)
    # exit()

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

if __name__ == "__main__":

    files = glob.glob(get_full_view_path("raw") + "/*.npy")

    times = []
    for file in files:
        save_file_path = file.replace("full_view/raw", "full_view/mb")
        print(save_file_path)
        t = time.time()
        recon = reconstruct(file).T
        times.append(time.time() - t)
        np.save(save_file_path, recon)
        print(times)
    times = np.asarray(times)
    print(np.mean(times[1:]), np.std(times[1:]))
