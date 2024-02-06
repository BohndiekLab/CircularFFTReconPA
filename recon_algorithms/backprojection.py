import patato as pat
import numpy as np
from patato.io.ipasc.read_ipasc import IPASCInterface
from utils.constants import *
import glob


def reconstruct(path, sound_speed=1488):
    pa_data = pat.PAData(IPASCInterface(PATH_IPASC_FILE))
    time_factor = 1
    detector_factor = 1
    preproc = pat.DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor,
                                          hilbert=False, lp_filter=7e6, hp_filter=5e3,
                                          irf=False)
    patato_recon = pat.ReferenceBackprojection(field_of_view=(0.032, 0.032, 0.032), n_pixels=(300, 1, 300))

    ts = pa_data.get_time_series()
    time_series = np.load(path)
    ts.raw_data = np.reshape(time_series, (1, 1, 256, -1))
    new_t1, d1, _ = preproc.run(ts, pa_data)
    recon, _, _ = patato_recon.run(new_t1, pa_data, sound_speed, **d1)
    return (np.asarray(np.squeeze(recon.raw_data))).copy()


# reconstruct sim
for file in glob.glob(PATH_CAL_SIM + "/*.npy"):
    save_file_path = file.replace("sim", "recons/bp/sim")
    recon = reconstruct(file).T
    np.save(save_file_path, recon)

# reconstruct sim_raw
for file in glob.glob(PATH_CAL_SIM_RAW + "/*.npy"):
    save_file_path = file.replace("sim_raw", "recons/bp/sim_raw")
    recon = reconstruct(file).T
    np.save(save_file_path, recon)

# reconstruct exp
for file in glob.glob(PATH_CAL_EXP + "/*.npy"):
    save_file_path = file.replace("exp", "recons/bp/exp")
    recon = reconstruct(file).T
    np.save(save_file_path, recon)
