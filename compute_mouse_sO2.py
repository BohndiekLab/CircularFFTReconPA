from utils.constants import get_recon_path, PATH_BASE
import numpy as np
import matplotlib.pyplot as plt
from utils.linear_unmixing import linear_unmixing
from matplotlib_scalebar.scalebar import ScaleBar
from utils.histogram_colourbar import add_histogram_colorbar
import string
import nrrd

SPACING = 0.10666666667

ALGORITHMS = ["BP", "MB", "BP", "BP", "BP"]
NAMES = ["Backprojection",
         "Model-based",
         "Time Reversal",
         "Iterative TR",
         "Circular FFT"]

SCANS = ["3", "5", "7", "9", "15", "17", "21", "23"]

MASK_LABELS = {
    1: "BACKGROUND",
    2: "BODY",
    3: "SPLEEN",
    4: "KIDNEY",
    5: "SPINE",
    6: "ARTERY"
}


selection = np.s_[25:275, 50:300]
res = dict()
for algo_idx in range(5):
    res[NAMES[algo_idx]] = dict()
    for idx in range(1, 7):
        res[NAMES[algo_idx]][MASK_LABELS[idx]] = []

for scan in SCANS:

    lbl, _ = nrrd.read(f"{PATH_BASE}/data/mice/mask/Scan_{scan}-labels.nrrd")
    labels = np.ones((300, 300))
    labels[6:-6, 6:-6] = lbl[:, :, 0]
    labels = np.fliplr(labels)[selection]

    for a_idx, algo in enumerate(ALGORITHMS):
        data = np.load(get_recon_path("mice", "exp", algo) + f"/Scan_{scan}.npy").T
        lu = np.squeeze(linear_unmixing(np.abs(data), [700, 730, 750, 760, 770, 800, 820, 840, 850, 880])) * 100
        lu = lu[selection]
        for idx in range(1, 7):
            res[NAMES[a_idx]][MASK_LABELS[idx]].append(np.mean(lu[labels == idx]))


def res_format(numbers: list):
    return f"{np.mean(numbers):.0f}$\pm${np.std(numbers):.0f}"


print(
f"\\begin{{table}}[!ht]\n"
    f"\t\\centering\n"
    f"\t\\begin{{tabular}}{{llllll}}\n"
        f"\t\t & \\textbf{{{ALGORITHMS[0]}}} & \\textbf{{{ALGORITHMS[1]}}} & \\textbf{{{ALGORITHMS[2]}}} & \\textbf{{{ALGORITHMS[3]}}} & \\textbf{{{ALGORITHMS[4]}}} \\\\ \\hline\n"
        f"\t\t\\\\\n"
        f"\t\t\\textbf{{{MASK_LABELS[2]}}} & {res_format(res[NAMES[0]][MASK_LABELS[2]])} & {res_format(res[NAMES[1]][MASK_LABELS[2]])} & {res_format(res[NAMES[2]][MASK_LABELS[2]])} & {res_format(res[NAMES[3]][MASK_LABELS[2]])} & {res_format(res[NAMES[4]][MASK_LABELS[2]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[3]}}} & {res_format(res[NAMES[0]][MASK_LABELS[3]])} & {res_format(res[NAMES[1]][MASK_LABELS[3]])} & {res_format(res[NAMES[2]][MASK_LABELS[3]])} & {res_format(res[NAMES[3]][MASK_LABELS[3]])} & {res_format(res[NAMES[4]][MASK_LABELS[3]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[4]}}} & {res_format(res[NAMES[0]][MASK_LABELS[4]])} & {res_format(res[NAMES[1]][MASK_LABELS[4]])} & {res_format(res[NAMES[2]][MASK_LABELS[4]])} & {res_format(res[NAMES[3]][MASK_LABELS[4]])} & {res_format(res[NAMES[4]][MASK_LABELS[4]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[5]}}} & {res_format(res[NAMES[0]][MASK_LABELS[5]])} & {res_format(res[NAMES[1]][MASK_LABELS[5]])} & {res_format(res[NAMES[2]][MASK_LABELS[5]])} & {res_format(res[NAMES[3]][MASK_LABELS[5]])} & {res_format(res[NAMES[4]][MASK_LABELS[5]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[6]}}} & {res_format(res[NAMES[0]][MASK_LABELS[6]])} & {res_format(res[NAMES[1]][MASK_LABELS[6]])} & {res_format(res[NAMES[2]][MASK_LABELS[6]])} & {res_format(res[NAMES[3]][MASK_LABELS[6]])} & {res_format(res[NAMES[4]][MASK_LABELS[6]])} \\\\ \n"
    f"\t\\end{{tabular}}\n"
    f"\\caption{{Results on sO$_2$.}}"
f"\\end{{table}}\n"
)
