# Step 1 for implementing feedback from review process.
# Reviewers were generally unsure why digital twins are a good idea.
# They also claimed that it is a common practice in the field to do the image comparison this way.

# Selling point for digital twins: we have corresponding simulated and experimental data

from utils.constants import get_raw_path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy.stats import pearsonr
from scipy.stats import wilcoxon

sim_path = get_raw_path(data_set="testing", data="sim")
sim_raw_path = get_raw_path(data_set="testing", data="sim_raw")
exp_path = get_raw_path(data_set="testing", data="exp")

corr_sim = []
corr_sim_raw = []
rmse_errors_sim = []
rmse_errors_sim_raw = []

for phantom_file in glob.glob(sim_path + "/*.npy"):
    phantom_name = Path(phantom_file).stem
    print(phantom_name)

    sim_phantom = np.load(f"{sim_path}/{phantom_name}.npy")[:, 600:1600]
    sim_raw_phantom = np.load(f"{sim_raw_path}/{phantom_name}.npy")[:, 600:1600] / 10
    exp_phantom = np.load(f"{exp_path}/{phantom_name}.npy")[:, 600:1600]
    exp_phantom[exp_phantom == 0] = 0.001

    corr, _ = pearsonr(sim_phantom.reshape((-1,)), exp_phantom.reshape((-1,)))
    corr_sim.append(corr)

    corr, _ = pearsonr(sim_raw_phantom.reshape((-1,)), exp_phantom.reshape((-1,)))
    corr_sim_raw.append(corr)

    rmse_errors_sim.append(np.mean((sim_phantom - exp_phantom)**2))
    rmse_errors_sim_raw.append(np.mean((sim_raw_phantom - exp_phantom) ** 2))


corr_stat = wilcoxon(np.asarray(corr_sim).reshape((-1,)), np.asarray(corr_sim_raw).reshape((-1,)))
corr_p_value = corr_stat.pvalue

rsme_stat = wilcoxon(np.asarray(rmse_errors_sim).reshape((-1,)), np.asarray(rmse_errors_sim_raw).reshape((-1,)))
rsme_p_value = rsme_stat.pvalue

print(f"SIM")
print(np.median(corr_sim), corr_p_value)
print(np.sqrt(np.median(rmse_errors_sim)), rsme_p_value)

print("SIM_RAW")
print(np.median(corr_sim_raw))
print(np.sqrt(np.median(rmse_errors_sim_raw)))

EXAMPLE_PHANTOM = "P.5.23_800"
EXAMPLE_DETECTOR_1 = 128

sim_phantom = np.load(f"{get_raw_path('testing', 'sim')}/{EXAMPLE_PHANTOM}.npy")[:, 650:1550]
sim_raw_phantom = np.load(f"{get_raw_path('testing', 'sim_raw')}/{EXAMPLE_PHANTOM}.npy")[:, 650:1550] / 10
exp_phantom = np.load(f"{get_raw_path('testing', 'exp')}/{EXAMPLE_PHANTOM}.npy")[:, 650:1550]

fig = plt.figure(figsize=(6, 4.5), layout="constrained")

gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[0.8, 1], wspace=0.0, hspace=0.0)

ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax_top = fig.add_subplot(gs[0, :])

SIZE = 10

ax1.boxplot([corr_sim_raw, corr_sim], showfliers=False, widths=0.5, labels=["naïve", "calibrated"])
xses = np.ones_like(corr_sim_raw) + (np.random.random(np.shape(corr_sim_raw))*0.4-0.2)
selection = ((corr_sim_raw > np.percentile(corr_sim_raw, 5)) &
             (corr_sim_raw < np.percentile(corr_sim_raw, 95)))
ax1.scatter(xses[selection], np.asarray(corr_sim_raw)[selection], c="black", alpha=0.5, s=SIZE)
selection = ((corr_sim > np.percentile(corr_sim, 5)) &
             (corr_sim < np.percentile(corr_sim, 95)))
ax1.scatter(xses[selection]+1, np.asarray(corr_sim)[selection], c="blue", alpha=0.5, s=SIZE)
ax1.spines[["right", "top"]].set_visible(False)
ax1.set_ylabel("Correlation Coefficient [-1, 1]")

# Define bar position
y_max = max(max(corr_sim_raw), max(corr_sim))  # Highest point in the data
y_min = min(min(corr_sim_raw), min(corr_sim))
y_bar = y_max + (y_max - y_min) * 0.05  # Position slightly above data
y_text = y_bar  # Position for text

# Draw the bar
ax1.plot([1.0, 2.0], [y_bar, y_bar], color='black')  # Horizontal line
ax1.text(1.5, y_text, f"***", ha='center', fontsize=12)
rmse_errors_sim_raw = np.sqrt(rmse_errors_sim_raw)
rmse_errors_sim = np.sqrt(rmse_errors_sim)
ax2.boxplot([rmse_errors_sim_raw, rmse_errors_sim], showfliers=False, widths=0.5, labels=["naïve", "calibrated"])
selection = ((rmse_errors_sim_raw > np.percentile(rmse_errors_sim_raw, 5)) &
             (rmse_errors_sim_raw < np.percentile(rmse_errors_sim_raw, 95)))
ax2.scatter(xses[selection], np.asarray(rmse_errors_sim_raw)[selection], c="black", alpha=0.5, s=SIZE)
selection = ((rmse_errors_sim > np.percentile(rmse_errors_sim, 5)) &
             (rmse_errors_sim < np.percentile(rmse_errors_sim, 95)))
ax2.scatter(xses[selection]+1, np.asarray(rmse_errors_sim)[selection], c="blue", alpha=0.5, s=SIZE)
ax2.spines[["right", "top"]].set_visible(False)
ax2.set_ylabel("RMSE [a.u.]")

# Define bar position
y_max = max(np.percentile(rmse_errors_sim_raw, 95), np.percentile(rmse_errors_sim, 95))  # Highest point in the data
y_min = min(min(rmse_errors_sim_raw), min(rmse_errors_sim))
y_bar = y_max + (y_max - y_min) * 0.2  # Position slightly above data
y_text = y_bar  # Position for text

# Draw the bar
ax2.plot([1.0, 2.0], [y_bar, y_bar], color='black')  # Horizontal line
ax2.text(1.5, y_text, f"***", ha='center', fontsize=12)

ax_top.plot(sim_phantom[EXAMPLE_DETECTOR_1], c="blue", label="calibrated simulation")
ax_top.plot(exp_phantom[EXAMPLE_DETECTOR_1], c="green", label="experiment")
ax_top.spines[["right", "top"]].set_visible(False)
ax_top.set_ylabel("Acoustic Pressure [a.u.]")
ax_top.set_xlabel("Time series point")
ax_top.legend()

ax1.text(0.03, 0.87, "B", transform=ax1.transAxes,
           size=24, weight='bold', color="black")
ax2.text(0.03, 0.87, "C", transform=ax2.transAxes,
           size=24, weight='bold', color="black")
ax_top.text(0.03, 0.87, "A", transform=ax_top.transAxes,
           size=24, weight='bold', color="black")

plt.savefig("show_forward_model_accuracy_vertical_2.pdf")
plt.close()