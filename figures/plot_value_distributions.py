import matplotlib.pyplot as plt
from utils.constants import get_recon_path
import numpy as np
import glob

# data = "exp"
# normalise = True
algorithms = {
    "BP": "Delay And Sum",
    "BPH": "Filtered Backprojection",
    "FFT": "Circular FFT",
    "TR": "Time Reversal",
    "ITTR": "Iterative Time Reversal",
    "MB": "Model-based",
}

fig = plt.figure(figsize=(12, 5), layout="constrained")
subfigs = fig.subfigures(2, 2, wspace=0.05, hspace=0.05)


def populate_subplot(data, normalise, subfig):
    axes = subfig.subplots(1, len(algorithms.keys()))
    for idx, key in enumerate(algorithms.keys()):
        files = glob.glob(get_recon_path(data_set="testing", data=data, algorithm=key) + "/*.npy")
        values = []
        for file in files:
            values.append(np.load(file))
        values = np.asarray(values).reshape((-1, ))
        values = np.random.choice(values.reshape(-1,), 100000)
        if normalise:
            intercept = np.load(f"../calibrate/cal_{key}_{data}.npz")["intercept"]
            slope = np.load(f"../calibrate/cal_{key}_{data}.npz")["slope"]
            values = intercept + slope * values
        label = algorithms[key].replace(" ", "\n")
        axes[idx].violinplot(values, positions=[0], showextrema=False, points=50, widths=0.8)
        axes[idx].boxplot(values, positions=[0], labels=[label], showfliers=False, widths=0.8)
        axes[idx].spines[["top", "right", "bottom"]].set_visible(False)
        if normalise:
            axes[idx].set_ylim(-100, 900)
        else:
            axes[idx].set_ylim(np.nanpercentile(values, 0.2), np.nanpercentile(values, 99.5))


populate_subplot("sim_raw", False, subfigs[0, 0])
populate_subplot("sim_raw", True, subfigs[0, 1])
populate_subplot("exp", False, subfigs[1, 0])
populate_subplot("exp", True, subfigs[1, 1])

subfigs[0, 0].suptitle("$\\bf{A.}$ Simulated data before normalisation", ha="left", x=0)
subfigs[0, 1].suptitle("$\\bf{B.}$ Simulated data after normalisation", ha="left", x=0)
subfigs[1, 0].suptitle("$\\bf{C.}$ Experimental data before normalisation", ha="left", x=0)
subfigs[1, 1].suptitle("$\\bf{D.}$ Experimental data after normalisation", ha="left", x=0)

plt.show()




plt.show()
plt.close()
