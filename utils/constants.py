PATH_BASE = "F:/fft_recon_project/"


def get_path(data_set=None, data=None):
    if data_set is None:
        data_set = "calibration"
    if data is None:
        data = "exp"

    return f"{PATH_BASE}/data/{data_set}/{data}/"


def get_recon_path(data_set=None, data=None, algorithm=None):
    if data_set is None:
        data_set = "calibration"
    if data is None:
        data = "exp"
    if algorithm is None:
        algorithm = "bp"

    return f"{PATH_BASE}/data/{data_set}/recons/{algorithm}/{data}/"


PATH_IPASC_FILE = f"{PATH_BASE}/ipasc_file.hdf5"
