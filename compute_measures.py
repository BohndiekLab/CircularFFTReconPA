from calibrate.calibrate import calibrate_to_p0
from utils.load_data import load_all_recons, load_all
import numpy as np
import matplotlib.pyplot as plt
from utils.segmentation import get_coupling_medium_segmentation, get_unit_circle
from quality_control.measures import StructuralSimilarityIndex, MedianAbsoluteError, \
    JensenShannonDivergence, Sharpness, SharpnessHaarWaveletSparsity, SharpnessGradientSparsity
from quality_control.create_metric_window import apply_window_function
from matplotlib.patches import Rectangle

ALGORITHMS = ["BP", "BPH", "MB", "TR", "ITTR", "FFT"]
NAMES = ["Backprojection", "BP + Hilbert",
         "Model-based reconstruction",
         "Time reversal", "Iterative time reversal",
         "Circular FFT"]
data_source = "sim"

gt = load_all("testing", "p0")
water_segmentations = np.zeros_like(gt)
for idx in range(len(gt)):
    water_segmentations[idx] = get_coupling_medium_segmentation(gt[idx])
water_segmentations = water_segmentations.astype(bool)
res = dict()


for a_idx, algo in enumerate(ALGORITHMS):
    res[algo] = dict()
    print(algo)
    slope, intercept, r = calibrate_to_p0(algo, data_source)
    all_data = load_all_recons("testing", data_source, algo)
    images = intercept + slope * all_data

    print("\tR")
    res[algo]["R"] = r
    print("\tMAE")
    gt[~water_segmentations] = np.nan
    images[~water_segmentations] = np.nan
    res[algo]["MAE"] = MedianAbsoluteError(images, gt)
    print("\tSSIM")
    gt[~water_segmentations] = 0
    images[~water_segmentations] = 0
    res[algo]["SSIM"] = StructuralSimilarityIndex(images, gt)
    print("\tJSD")
    gt[~water_segmentations] = np.nan
    images[~water_segmentations] = np.nan
    res[algo]["JSD"] = JensenShannonDivergence(images.reshape((len(images), -1)), gt.reshape((len(images), -1)))
    print("\tSHRP")
    gt[~water_segmentations] = 0
    images[~water_segmentations] = 0
    res[algo]["SHRP"] = SharpnessHaarWaveletSparsity(images)

SHRP_SCALE = 1

print(
f"\\begin{{table}}[!ht]\n"
    f"\t\\centering\n"
    f"\t\\begin{{tabular}}{{llllll}}\n"
        f"\t\t & \\textbf{{{ALGORITHMS[0]}}} & \\textbf{{{ALGORITHMS[1]}}} & \\textbf{{{ALGORITHMS[2]}}} & \\textbf{{{ALGORITHMS[3]}}} & \\textbf{{{ALGORITHMS[4]}}} \\\\ \\hline\n"
        f"\t\t\\\\\n"
        f"\t\t\\textbf{{R}} & {res[ALGORITHMS[0]]['R']:.2f} & {res[ALGORITHMS[1]]['R']:.2f} & {res[ALGORITHMS[2]]['R']:.2f} & {res[ALGORITHMS[3]]['R']:.2f} & {res[ALGORITHMS[4]]['R']:.2f} \\\\ \n"
        f"\t\t\\textbf{{MAE}} & {res[ALGORITHMS[0]]['MAE']:.0f} & {res[ALGORITHMS[1]]['MAE']:.0f} & {res[ALGORITHMS[2]]['MAE']:.0f} & {res[ALGORITHMS[3]]['MAE']:.0f} & {res[ALGORITHMS[4]]['MAE']:.0f} \\\\ \n"
        f"\t\t\\textbf{{SSIM}} & {res[ALGORITHMS[0]]['SSIM']:.2f} & {res[ALGORITHMS[1]]['SSIM']:.2f} & {res[ALGORITHMS[2]]['SSIM']:.2f} & {res[ALGORITHMS[3]]['SSIM']:.2f} & {res[ALGORITHMS[4]]['SSIM']:.2f} \\\\ \n"
        f"\t\t\\textbf{{JSD}} & {res[ALGORITHMS[0]]['JSD']:.2f} & {res[ALGORITHMS[1]]['JSD']:.2f} & {res[ALGORITHMS[2]]['JSD']:.2f} & {res[ALGORITHMS[3]]['JSD']:.2f} & {res[ALGORITHMS[4]]['JSD']:.2f} \\\\ \n"
        f"\t\t\\textbf{{SHRP}} & {res[ALGORITHMS[0]]['SHRP']/SHRP_SCALE:.2f} & {res[ALGORITHMS[1]]['SHRP']/SHRP_SCALE:.2f} & {res[ALGORITHMS[2]]['SHRP']/SHRP_SCALE:.2f} & {res[ALGORITHMS[3]]['SHRP']/SHRP_SCALE:.2f} & {res[ALGORITHMS[4]]['SHRP']/SHRP_SCALE:.2f} \\\\ \n"
    f"\t\\end{{tabular}}\n"
    f"\\caption{{Results on {data_source}.}}"
f"\\end{{table}}\n"
)
