import numpy as np
from matplotlib.patches import Rectangle


def add_histogram_colorbar(ax, data, width=None, height=None, colormap=None, label=None,
                           vmin=None, vmax=None, min_label=None, max_label=None):

    ydim, xdim = np.shape(data)

    if width is None:
        width = xdim / 3
    if height is None:
        height = ydim / 10
    if colormap is None:
        colormap = "viridis"
    if label is None:
        label = ""
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    if min_label is None:
        min_label = "min"
    if max_label is None:
        max_label = "max"

    bar_height = int(20 / 300 * ydim)

    baseline = ydim - 25
    leftshift = 15
    outline_color = "black"

    hist, _ = np.histogram(np.reshape(data, (-1, )), bins=50, range=(vmin, vmax))
    bins = np.linspace(0, 100, 51)
    ax.plot([xdim - width - leftshift, xdim - leftshift],
            [baseline, baseline], c=outline_color, linewidth=1)
    ax.stairs(baseline - (hist / np.max(hist) * height),
              (xdim - leftshift - width + bins / 100 * width),
              baseline=baseline, fill=True, color="lightgray")
    ax.stairs(baseline - (hist / np.max(hist) * height),
              (xdim - leftshift - width + bins / 100 * width),
              baseline=baseline, fill=False, color=outline_color, linewidth=1)
    x_points = np.asarray(xdim - leftshift - width + bins / 100 * width)

    for i in np.arange(0, bar_height, 2):
        ax.scatter(x_points, np.ones_like(x_points) * baseline + 2 + i,
                   c=((x_points - np.min(x_points)) / (np.max(x_points) - np.min(x_points))),
                   s=2, marker="s", cmap=colormap)

    ax.add_artist(Rectangle((xdim - leftshift - width - 2, baseline), width + 4,
                            bar_height + 2, fill=False, edgecolor="black"))

    ax.text(xdim - leftshift - width, baseline + bar_height - 2, min_label, color="white", fontsize=8, ha="left")
    ax.text(xdim - leftshift - width / 2, baseline + bar_height - 2, label, color="white", fontsize=8, ha="center")
    ax.text(xdim - leftshift, baseline + bar_height - 2, max_label, color="white", fontsize=8, ha="right")
