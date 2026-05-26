from __future__ import annotations

from pathlib import Path

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

# Shared colormap for every heatmap stage (mako: dark teal -> light yellow).
# Single source of truth so cosine / rsa / judge stay visually consistent.
DIVERGING_CMAP = sns.color_palette("mako", as_cmap=True)


def save_figure(fig, png_path: Path | str, **savefig_kwargs) -> Path:
    """Save a matplotlib figure as both PNG and SVG with the same stem."""
    png_path = Path(png_path)
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, **savefig_kwargs)
    fig.savefig(svg_path, **savefig_kwargs)
    return png_path


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