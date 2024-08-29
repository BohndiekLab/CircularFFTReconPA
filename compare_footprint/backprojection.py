import patato as pat
import numpy as np
from patato.io.ipasc.read_ipasc import IPASCInterface
from utils.constants import *
import glob
import time
import tracemalloc

data_source = "testing"


def reconstruct(path, sound_speed=1488):
    pa_data = pat.PAData(IPASCInterface(PATH_IPASC_FILE))
    time_factor = 1
    detector_factor = 1
    preproc = pat.DefaultMSOTPreProcessor(time_factor=time_factor, detector_factor=detector_factor,
                                          hilbert=False, lp_filter=None, hp_filter=None,
                                          irf=False)
    patato_recon = pat.ReferenceBackprojection(field_of_view=(0.032, 0.032, 0.032), n_pixels=(300, 1, 300))

    ts = pa_data.get_time_series()
    time_series = np.load(path)
    ts.raw_data = np.reshape(time_series, (1, 1, 256, -1))
    new_t1, d1, _ = preproc.run(ts, pa_data)
    recon, _, _ = patato_recon.run(new_t1, pa_data, sound_speed, **d1)
    return (np.asarray(np.squeeze(recon.raw_data))).copy()


times = []
# reconstruct sim_raw
for file in glob.glob(get_raw_path(data_source, "sim_raw") + "/*.npy")[0:1]:
    # Start tracing memory allocations
    tracemalloc.start()
    print(file)
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

