PATH_BASE = "F:/fft_recon_project/"


def get_raw_path(data_set=None, data=None):
    if data_set is None:
        data_set = "calibration"
    if data is None:
        data = "exp"
    return f"{PATH_BASE}/data/{data_set}/raw/{data}/"


def get_p0_path(data_set=None):
    if data_set is None:
        data_set = "calibration"
    return f"{PATH_BASE}/data/{data_set}/p0/"


def get_mouse_recon_path(algorithm=None):
    if algorithm is None:
        algorithm = "bp"
    return f"{PATH_BASE}/data/mice/recons/{algorithm}/"


def get_recon_path(data_set=None, data=None, algorithm=None):
    if data_set is None:
        data_set = "calibration"
    if data is None:
        data = "exp"
    if algorithm is None:
        algorithm = "bp"

    return f"{PATH_BASE}/data/{data_set}/recons/{algorithm}/{data}/"


PATH_IPASC_FILE = f"{PATH_BASE}/ipasc_file.hdf5"
