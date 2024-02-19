from calibrate.calibrate import calibrate_to_p0
from utils.load_data import load_all_recons, load_all
import numpy as np
import matplotlib.pyplot as plt
from utils.segmentation import get_coupling_medium_segmentation
from quality_control.measures import StructuralSimilarityIndex, JensenShannonDivergence
from quality_control.create_metric_window import apply_window_function
from matplotlib.patches import Rectangle

ALGORITHMS = ["BP", "MB", "TR", "ITTR", "FFT"]
NAMES = ["Backprojection",
         "Model-based reconstruction",
         "Time reversal", "Iterative time reversal",
         "Circular FFT"]
data_source = "exp"
EXAMPLE_IMAGE_IDX = 42

fig, axes = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")

axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]

gt = load_all("calibration", "p0")
print(np.min(gt), np.max(gt))
water_segmentation = get_coupling_medium_segmentation(gt[EXAMPLE_IMAGE_IDX])
gt_img = gt[EXAMPLE_IMAGE_IDX].copy()
# gt_img[~water_segmentation] = np.nan

plt.suptitle(f"Comparing reconstructions on {data_source} data.", fontweight="bold")

axes[0].imshow(gt_img.T, vmin=np.nanmin(gt_img), vmax=np.nanmax(gt_img))
axes[0].axis("off")
axes[0].text(50, 10, "Simulated p$_0$")

res = dict()


def mae(x, y):
    return np.nanmedian(np.abs(x - y))


for a_idx, algo in enumerate(ALGORITHMS):
    res[algo] = dict()
    print(algo)
    slope, intercept, r = calibrate_to_p0(algo, data_source)
    all_data = load_all_recons("calibration", data_source, algo)
    print(np.min(all_data), np.max(all_data))
    images = intercept + slope * all_data

    res[algo]["R"] = r
    res[algo]["MAE"] = mae(images, gt)
    res[algo]["SSIM"] = StructuralSimilarityIndex(images, gt)
    res[algo]["JSD"] = JensenShannonDivergence(images.reshape((len(images), -1)), gt.reshape((len(images), -1)))

    sample_image = images[EXAMPLE_IMAGE_IDX].copy()
    sample_error_window = apply_window_function(gt_img, sample_image, 15, StructuralSimilarityIndex)
    # sample_image[~water_segmentation] = np.nan
    axes[a_idx + 1].imshow(sample_image.T, vmin=np.nanmin(gt_img), vmax=np.nanmax(gt_img))
    axes[a_idx + 1].axis("off")
    axes[a_idx + 1].add_patch(Rectangle((10, 0), 280, 22, color="grey", fill=True))
    axes[a_idx + 1].text(25, 20, NAMES[a_idx])


def print_results(axis, results, algorithms, first=True):
    axis.axis("off")

    algo_xs = np.asarray([0.25, 0.5, 0.75])
    if first:
        axis.text(0, 0.8, "R")
        axis.text(0, 0.6, "MAE")
        axis.text(0, 0.4, "SSIM")
        axis.text(0, 0.2, "JSD")
    else:
        algo_xs -= 0.25

    axis.text(algo_xs[0], 1.0, algorithms[0])
    axis.text(algo_xs[1], 1.0, algorithms[1])
    if len(algorithms) == 3:
        axis.text(algo_xs[2], 1.0, algorithms[2])

    for measure, measure_y in zip(["R", "MAE", "SSIM", "JSD"], [0.8, 0.6, 0.4, 0.2]):
        for algo, algo_x in zip(algorithms, algo_xs):
            axis.text(algo_x, measure_y, f"{results[algo][measure]:.2f}")


print_results(axes[-2], res, ["BP", "MB", "TR"])
print_results(axes[-1], res, ["ITTR", "FFT"], first=False)

plt.savefig("Figure2.png", dpi=300)
plt.show()
