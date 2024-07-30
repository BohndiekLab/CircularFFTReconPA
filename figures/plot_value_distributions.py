import matplotlib.pyplot as plt
import numpy as np

data = "exp"
normalise = True
algorithms = {
    "BP": "Delay And Sum",
    "BPH": "Filtered Backprojection",
    "FFT": "Circular FFT",
    "TR": "Time Reversal",
    "ITTR": "Iterative Time Reversal",
    "MB": "Model-based",
}

all_values = []
fig, axes = plt.subplots(1, len(algorithms.keys()), layout="constrained", figsize=(8, 2))
for idx, key in enumerate(algorithms.keys()):
    values = np.load(f"../calibrate/cal_{key}_{data}.npz")["target"]
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
        axes[idx].set_ylim(np.percentile(values, 0.2), np.percentile(values, 99.5))

plt.show()




plt.show()
plt.close()
