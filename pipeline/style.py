from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DPI = 250
TITLE_WEIGHT = "bold"
TITLE_FONTSIZE = 22
AXIS_FONTSIZE = 14


COLORS = {
    "lima": "#E6334C",
    "nyc": "#1E88E5",
    "vlm": "#29B66C",
    "human_lima": "#E6334C",
    "human_nyc": "#1E88E5",
}


def apply_style() -> None:
    # plt.rcParams.update({
    #     "figure.dpi": BASE_DPI,
    #     #"axes.titlesize": TITLE_FONTSIZE,
    #     #"axes.labelsize": AXIS_FONTSIZE,
    #     "legend.fontsize": 12,
    #     "xtick.labelsize": 11,
    #     "ytick.labelsize": 11,
    #     "font.family": "sans-serif",
    # })
    pass


def reset_style() -> None:
    plt.rcParams.update(plt.rcParamsDefault)