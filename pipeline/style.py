from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize

from pipeline.config import (
    HEATMAP_BLEND_ALPHA,
    HEATMAP_BLEND_ENABLED,
    HEATMAP_BLEND_RES,
    HEATMAP_BLEND_WIDTH,
)

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

# Distinct filled markers; recycled within each agent group (VLM / human_lima / human_nyc).
# Color disambiguates between groups, so the same marker shape across groups is fine.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<", ">", "d", "H", "8"]


def build_agent_markers(agents) -> dict:
    """Map each agent to a stable marker. VLMs / human_lima / human_nyc cycle MARKERS independently."""
    import re

    def _natural_key(name: str) -> list:
        return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]

    vlms, humans_lima, humans_nyc = [], [], []
    for a in dict.fromkeys(agents):
        al = a.lower()
        if "human" in al and "lima" in al:
            humans_lima.append(a)
        elif "human" in al and "nyc" in al:
            humans_nyc.append(a)
        else:
            vlms.append(a)

    mapping = {}
    for group in (sorted(vlms, key=_natural_key), sorted(humans_lima, key=_natural_key), sorted(humans_nyc, key=_natural_key)):
        for i, agent in enumerate(group):
            mapping[agent] = MARKERS[i % len(MARKERS)]
    return mapping

# Shared colormap for every heatmap stage (mako: dark teal -> light yellow).
# Single source of truth so cosine / rsa / judge stay visually consistent.
DIVERGING_CMAP =  sns.cubehelix_palette(start=0, rot=0.2,reverse=False,as_cmap=True) #sns.color_palette("inferno", as_cmap=True)  #sns.color_palette("Greys",as_cmap=True) ##sns.cubehelix_palette(as_cmap=True) #.color_palette("rocket_r", as_cmap=True)


# Fallback inner-grid style (used only when blending is disabled), shared by every
# heatmap so the white separator lines render consistently across PNG and SVG/PDF.
HEATMAP_LINEWIDTHS = 0.5
HEATMAP_LINECOLOR = "white"


def blended_grid(data, ax, cmap=None, vmin=None, vmax=None, *,
                 width=HEATMAP_BLEND_WIDTH, res=HEATMAP_BLEND_RES,
                 alpha=HEATMAP_BLEND_ALPHA, zorder=2, mesh=None):
    """Draw the tri-point (A -> white -> B) seams between heatmap cells.

    Each internal border is a band perpendicular to it: cell A's color on one side,
    a static white middle, cell B's color on the other. Rendered as a single RGBA
    raster layer (transparent inside the cells), so the separators look identical in
    PNG and SVG/PDF without inflating the vector file.

    Cell colors are read from the already-rendered ``QuadMesh`` when available, so the
    seams match the heatmap exactly even when it uses ``center=``. Falls back to
    ``cmap``/``vmin``/``vmax`` mapping otherwise.
    """
    data = np.asarray(data, float)
    nrows, ncols = data.shape
    white = np.array([1.0, 1.0, 1.0, 1.0])

    # Cell colors — prefer the actually-rendered mesh (exact match, incl. `center`).
    cell = None
    if mesh is None and ax.collections:
        mesh = ax.collections[0]
    if mesh is not None:
        fc = np.asarray(mesh.get_facecolor())
        if fc.ndim == 2 and fc.shape[0] == nrows * ncols:
            cell = fc.reshape(nrows, ncols, 4).copy()
    if cell is None:                                   # fallback: map via cmap/norm
        cmap = cmap if cmap is not None else DIVERGING_CMAP
        cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        lo = np.nanmin(data) if vmin is None else vmin
        hi = np.nanmax(data) if vmax is None else vmax
        cell = np.asarray(cmap(Normalize(lo, hi)(np.ma.masked_invalid(data))))

    valid = ~np.isnan(data)

    H, W = nrows * res, ncols * res
    out = np.zeros((H, W, 4))                          # transparent overlay
    xs = (np.arange(W) + 0.5) / res                    # x in data coords [0, ncols]
    ys = (np.arange(H) + 0.5) / res                    # y in data coords [0, nrows]
    XX, YY = np.meshgrid(xs, ys)
    half = width / 2.0

    def tri(t, a_col, b_col):
        t = t[..., None]                               # t: 0 -> A, 0.5 -> white, 1 -> B
        lo = (1 - 2 * t) * a_col + (2 * t) * white
        hi = (2 - 2 * t) * white + (2 * t - 1) * b_col
        return np.where(t <= 0.5, lo, hi)

    nx = np.round(XX); dV = np.abs(XX - nx)            # nearest vertical border
    ny = np.round(YY); dH = np.abs(YY - ny)            # nearest horizontal border
    vmask = (dV <= half) & (nx >= 1) & (nx <= ncols - 1)
    hmask = (dH <= half) & (ny >= 1) & (ny <= nrows - 1)
    use_v = vmask & (~hmask | (dV <= dH))              # at corners: nearer border wins
    use_h = hmask & (~vmask | (dH < dV))

    rid = np.clip(YY.astype(int), 0, nrows - 1)
    cid = np.clip(XX.astype(int), 0, ncols - 1)

    if use_v.any():                                    # A = left cell, B = right cell
        cbp = np.clip(nx.astype(int), 0, ncols - 1)
        cap = np.clip(nx.astype(int) - 1, 0, ncols - 1)
        t = np.clip((XX - (nx - half)) / width, 0, 1)
        col = tri(t, cell[rid, cap], cell[rid, cbp])
        ok = use_v & valid[rid, cap] & valid[rid, cbp]
        out[ok] = col[ok]
    if use_h.any():                                    # A = top cell, B = bottom cell
        rbp = np.clip(ny.astype(int), 0, nrows - 1)
        rap = np.clip(ny.astype(int) - 1, 0, nrows - 1)
        t = np.clip((YY - (ny - half)) / width, 0, 1)
        col = tri(t, cell[rap, cid], cell[rbp, cid])
        ok = use_h & valid[rap, cid] & valid[rbp, cid]
        out[ok] = col[ok]

    out[..., 3] *= alpha
    ax.imshow(out, extent=(0, ncols, nrows, 0), origin="upper",
              interpolation="bilinear", zorder=zorder)


def styled_heatmap(data, ax, *, blend=None, blend_width=HEATMAP_BLEND_WIDTH,
                   blend_res=HEATMAP_BLEND_RES, blend_alpha=HEATMAP_BLEND_ALPHA,
                   **kwargs):
    """Single entry point for all heatmaps.

    Enforces the shared colormap and square cells, and draws the tri-point blended
    inner grid (see :func:`blended_grid`) so PNG and SVG/PDF look identical. Blending
    defaults come from ``pipeline.config`` and can be overridden per call via the
    ``blend`` / ``blend_width`` / ``blend_res`` / ``blend_alpha`` arguments. Any other
    kwarg is forwarded to ``sns.heatmap``.
    """
    kwargs.setdefault("cmap", DIVERGING_CMAP)
    kwargs.setdefault("square", True)

    use_blend = HEATMAP_BLEND_ENABLED if blend is None else blend
    if use_blend:
        kwargs.setdefault("linewidths", 0)             # the blended overlay draws the separators
        hm = sns.heatmap(data, ax=ax, **kwargs)
        blended_grid(data, ax, cmap=kwargs.get("cmap"),
                     vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"),
                     width=blend_width, res=blend_res, alpha=blend_alpha)
        return hm

    kwargs.setdefault("linewidths", HEATMAP_LINEWIDTHS)
    kwargs.setdefault("linecolor", HEATMAP_LINECOLOR)
    return sns.heatmap(data, ax=ax, **kwargs)


def save_figure(fig, png_path: Path | str, **savefig_kwargs) -> Path:
    """Save a matplotlib figure as both PNG and SVG with the same stem."""
    png_path = Path(png_path)
    svg_path = png_path.with_suffix(".svg")
    pdf_path = png_path.with_suffix(".pdf")
    #fig.savefig(png_path, **savefig_kwargs)
    fig.savefig(svg_path, **savefig_kwargs)
    fig.savefig(pdf_path, **savefig_kwargs)
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