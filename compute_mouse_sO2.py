from utils.constants import get_mouse_recon_path, PATH_BASE
import numpy as np
import matplotlib.pyplot as plt
from utils.linear_unmixing import linear_unmixing
from matplotlib_scalebar.scalebar import ScaleBar
from utils.histogram_colourbar import add_histogram_colorbar
import string
import nrrd

SPACING = 0.10666666667

ALGORITHMS = ["BP", "BPH", "MB", "TR_interp", "ITTR_interp", "FFT"]
NAMES = ["Delay And Sum", "Filtered Backprojection",
         "Model-based reconstruction",
         "Time reversal", "Iterative time reversal",
         "Circular FFT"]

SCANS = ["3", "5", "7", "9", "15", "17", "21", "23"]

MASK_LABELS = {
    1: "BACKGROUND",
    2: "BODY",
    3: "SPLEEN",
    4: "KIDNEY",
    5: "SPINE",
    6: "AORTA"
}

WAVELENGTH = np.asarray([700, 730, 750, 760, 770, 800, 820, 840, 850, 880])
WL_CUTOFF = 7


selection = np.s_[25:275, 50:300]
res = dict()
for algo_idx in range(len(ALGORITHMS)):
    res[NAMES[algo_idx]] = dict()
    for idx in range(1, 7):
        res[NAMES[algo_idx]][MASK_LABELS[idx]] = []

for scan in SCANS:
    print(scan)

    lbl, _ = nrrd.read(f"{PATH_BASE}/data/mice/mask/Scan_{scan}-labels.nrrd")
    labels = np.ones((300, 300))
    labels[6:-6, 6:-6] = lbl[:, :, 0]
    labels = np.fliplr(labels)[selection]

    for a_idx, algo in enumerate(ALGORITHMS):
        print("\t", algo)
        data = np.load(get_mouse_recon_path(algo) + f"/Scan_{scan}.npy")

        if algo != "BPH" and algo != "MB":
            data = np.asarray([data[i].T[selection] for i in range(len(data))])
        else:
            data = np.asarray([data[i][selection] for i in range(len(data))])

        data = data - np.mean(data)
        l1 = np.linalg.norm(np.reshape(data, (-1)), ord=1) / np.size(data)
        data = data + 4 * l1

        data[data <= 0] = np.nan

        data = data / data[5][None, :, :]

        lu = np.squeeze(linear_unmixing(data[:WL_CUTOFF], WAVELENGTH[:WL_CUTOFF])) * 100
        for idx in range(1, 7):
            if np.isnan(lu[labels == idx]).all():
                continue
            res[NAMES[a_idx]][MASK_LABELS[idx]].append(np.nanmean(lu[labels == idx]))


def res_format(numbers: list):
    return f"{np.nanmean(numbers):.0f}$\pm${np.nanstd(numbers):.0f}"


print(
f"\\begin{{table}}[!ht]\n"
    f"\t\\centering\n"
    f"\t\\begin{{tabular}}{{lllllll}}\n"
        f"\t\t & \\textbf{{DAS}} & \\textbf{{FBP}} & \\textbf{{MB}} & \\textbf{{TR}} & \\textbf{{ITTR}} &  \\textbf{{FFT}} \\\\ \\hline\n"
        f"\t\t\\\\\n"
        f"\t\t\\textbf{{{MASK_LABELS[2]}}} & {res_format(res[NAMES[0]][MASK_LABELS[2]])} & {res_format(res[NAMES[1]][MASK_LABELS[2]])} & {res_format(res[NAMES[2]][MASK_LABELS[2]])} & {res_format(res[NAMES[3]][MASK_LABELS[2]])} & {res_format(res[NAMES[4]][MASK_LABELS[2]])} & {res_format(res[NAMES[5]][MASK_LABELS[2]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[3]}}} & {res_format(res[NAMES[0]][MASK_LABELS[3]])} & {res_format(res[NAMES[1]][MASK_LABELS[3]])} & {res_format(res[NAMES[2]][MASK_LABELS[3]])} & {res_format(res[NAMES[3]][MASK_LABELS[3]])} & {res_format(res[NAMES[4]][MASK_LABELS[3]])} & {res_format(res[NAMES[5]][MASK_LABELS[3]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[4]}}} & {res_format(res[NAMES[0]][MASK_LABELS[4]])} & {res_format(res[NAMES[1]][MASK_LABELS[4]])} & {res_format(res[NAMES[2]][MASK_LABELS[4]])} & {res_format(res[NAMES[3]][MASK_LABELS[4]])} & {res_format(res[NAMES[4]][MASK_LABELS[4]])} & {res_format(res[NAMES[5]][MASK_LABELS[4]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[5]}}} & {res_format(res[NAMES[0]][MASK_LABELS[5]])} & {res_format(res[NAMES[1]][MASK_LABELS[5]])} & {res_format(res[NAMES[2]][MASK_LABELS[5]])} & {res_format(res[NAMES[3]][MASK_LABELS[5]])} & {res_format(res[NAMES[4]][MASK_LABELS[5]])} & {res_format(res[NAMES[5]][MASK_LABELS[5]])} \\\\ \n"
        f"\t\t\\textbf{{{MASK_LABELS[6]}}} & {res_format(res[NAMES[0]][MASK_LABELS[6]])} & {res_format(res[NAMES[1]][MASK_LABELS[6]])} & {res_format(res[NAMES[2]][MASK_LABELS[6]])} & {res_format(res[NAMES[3]][MASK_LABELS[6]])} & {res_format(res[NAMES[4]][MASK_LABELS[6]])} & {res_format(res[NAMES[5]][MASK_LABELS[6]])} \\\\ \n"
    f"\t\\end{{tabular}}\n"
    f"\\caption{{Results on sO$_2$.}}"
f"\\end{{table}}\n"
)
