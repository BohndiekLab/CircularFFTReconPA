from calibrate.calibrate import calibrate_to_p0
import numpy as np
from utils.constants import get_p0_path, get_recon_path
from utils.segmentation import get_coupling_medium_segmentation
from measures.measures import StructuralSimilarityIndex, MedianAbsoluteError, \
    JensenShannonDivergence, SharpnessHaarWaveletSparsity, HaarPSI
import glob
import json

ALGORITHMS = ["BP", "BPH", "MB", "TR_interp", "ITTR_interp", "FFT"]
NAMES = ["Delay And Sum", "Filtered Backprojection",
         "Model-based reconstruction",
         "Time reversal", "Iterative time reversal",
         "Circular FFT"]
data_source = "sim_raw"

files = glob.glob(get_p0_path(data_set="testing") + "/*.npy")
p0 = []
for file in files:
    p0.append(np.load(file))
gt = np.asarray(p0)
water_segmentations = np.zeros_like(gt)
for idx in range(len(gt)):
    water_segmentations[idx] = get_coupling_medium_segmentation(gt[idx])
water_segmentations = water_segmentations.astype(bool)

with open("measures/sharpness.json", "r+") as jsonfile:
    fwhm = json.load(jsonfile)

res = dict()

for a_idx, algo in enumerate(ALGORITHMS):
    res[algo] = dict()
    print(algo)
    slope, intercept, r = calibrate_to_p0(algo.replace("_interp", ""), data_source)
    files = glob.glob(get_recon_path(data_set="testing", data=data_source, algorithm=algo) + "/*.npy")
    values = []
    for file in files:
        values.append(np.load(file))
    all_data = np.asarray(values)
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
    print("\tHaarPSI")
    gt[~water_segmentations] = 0
    images[~water_segmentations] = 0
    res[algo]["HaarPSI"] = HaarPSI(images, gt)
    print("\tSPARSE")
    gt[~water_segmentations] = 0
    images[~water_segmentations] = 0
    res[algo]["SPARSE"] = SharpnessHaarWaveletSparsity(images)

    print("\tFWHM")
    res[algo]["FWHM"] = np.mean(fwhm[data_source][algo])


SHRP_SCALE = 1

table_string = f"\\begin{{table}}[!ht]\n\t\\centering\n\t\\begin{{tabular}}{{lllllll}}\n\t\t"
for i in range(len(ALGORITHMS)):
    table_string += f" & \\textbf{{{ALGORITHMS[i]}}}"
table_string += "\\\\ \\hline\n\t\t\\\\\n"
for measure in ["R", "MAE", "SSIM", "JSD", "HaarPSI", "SPARSE", "FWHM"]:
    table_string +=f"\t\t\\textbf{{{measure}}}"
    for i in range(len(ALGORITHMS)):
        table_string += f" & {res[ALGORITHMS[i]][measure]:.2f}"
    table_string += "\\\\ \n"
table_string += f"\t\\end{{tabular}}\n\\caption{{Results on {data_source}.}}\\end{{table}}\n"

print(table_string)
