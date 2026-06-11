"""Interactive cmap viewer for the RSA heatmap.

Opens a matplotlib window with sliders to dial in a diverging cmap (two-extreme hues)
suitable for RSA correlations (vmin=-1, vmax=1). Three families switchable via radio:
  * diverging_palette : sns.diverging_palette(h_neg, h_pos, s, l, sep)
  * cubehelix mirrored: mirror a cubehelix sequence to make it diverging
  * named diverging   : index into a curated list of named diverging palettes

Click "Print spec" to dump the current spec to stdout — paste it as DIVERGING_CMAP
in pipeline/style.py.

Usage:
    python scripts/cmap_interactive.py [region] [block]
    python scripts/cmap_interactive.py NYC 1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons, Slider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.utils.metrics import get_ordered_agents

NAMED_DIVERGING = [
    "vlag", "icefire",
    "coolwarm", "bwr", "seismic",
    "RdBu_r", "RdYlBu_r", "RdYlGn_r", "Spectral_r",
    "PiYG", "PRGn", "BrBG", "PuOr",
]


def load_rsa_matrix(region: str, block: int) -> np.ndarray:
    parquet_path = ROOT / "outputs" / "pipeline" / "rsa" / "rsa_correlations.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Run the rsa stage first; missing {parquet_path}")
    rsa_df = pd.read_parquet(parquet_path)
    df_block = rsa_df[(rsa_df["VIDEO_SECTOR"] == region) & (rsa_df["BLOCK"] == block)]
    if df_block.empty:
        raise ValueError(f"No RSA rows for region={region} block={block}")
    agent_order = get_ordered_agents(
        pd.concat([df_block["AGENT_I"], df_block["AGENT_J"]]).unique()
    )
    return (
        df_block.pivot(index="AGENT_I", columns="AGENT_J", values="CORRELATION")
        .reindex(index=agent_order, columns=agent_order)
        .to_numpy()
    )


def cubehelix_mirrored(start: float, rot: float, dark: float, light: float) -> LinearSegmentedColormap:
    """Diverging cmap built by mirroring a cubehelix sequence around its midpoint."""
    seq = sns.cubehelix_palette(start=start, rot=rot, dark=dark, light=light, n_colors=128)
    colors = list(seq[::-1]) + list(seq)
    return LinearSegmentedColormap.from_list("cubehelix_mirrored", colors)


def main(region: str = "NYC", block: int = 1) -> None:
    matrix = load_rsa_matrix(region, block)

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(left=0.06, right=0.75, bottom=0.40, top=0.93)

    # Initial cmap: classic diverging palette
    h_neg, h_pos = 220, 20
    s_init, l_init, sep_init = 75, 50, 10
    cmap0 = sns.diverging_palette(h_neg, h_pos, s=s_init, l=l_init, sep=sep_init, as_cmap=True)
    im = ax.imshow(matrix, cmap=cmap0, vmin=-1, vmax=1, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"RSA cmap viewer — {region}, Block {block}\n"
        f"[diverging_palette] h_neg={h_neg} h_pos={h_pos} s={s_init} l={l_init} sep={sep_init}",
        fontsize=11,
    )

    # Sliders
    sl_h_neg = Slider(plt.axes([0.10, 0.32, 0.60, 0.022]), "h_neg",     0, 360, valinit=h_neg,    valstep=1)
    sl_h_pos = Slider(plt.axes([0.10, 0.29, 0.60, 0.022]), "h_pos",     0, 360, valinit=h_pos,    valstep=1)
    sl_s     = Slider(plt.axes([0.10, 0.26, 0.60, 0.022]), "sat",       0, 100, valinit=s_init,   valstep=1)
    sl_l     = Slider(plt.axes([0.10, 0.23, 0.60, 0.022]), "light",     0, 100, valinit=l_init,   valstep=1)
    sl_sep   = Slider(plt.axes([0.10, 0.20, 0.60, 0.022]), "sep",       0, 50,  valinit=sep_init, valstep=1)
    sl_start = Slider(plt.axes([0.10, 0.15, 0.60, 0.022]), "ch start",  0.0, 3.0, valinit=0.5,  valstep=0.05)
    sl_rot   = Slider(plt.axes([0.10, 0.12, 0.60, 0.022]), "ch rot",   -2.0, 2.0, valinit=-0.5, valstep=0.05)
    sl_dark  = Slider(plt.axes([0.10, 0.09, 0.60, 0.022]), "ch dark",   0.0, 0.5, valinit=0.2,  valstep=0.02)
    sl_light = Slider(plt.axes([0.10, 0.06, 0.60, 0.022]), "ch light",  0.5, 1.0, valinit=0.95, valstep=0.02)
    sl_named = Slider(plt.axes([0.10, 0.02, 0.60, 0.022]), "named idx", 0, len(NAMED_DIVERGING) - 1, valinit=0, valstep=1)

    # Radio: family selector
    radio = RadioButtons(
        plt.axes([0.77, 0.16, 0.20, 0.16]),
        ("diverging_palette", "cubehelix mirrored", "named"),
    )

    # Buttons
    btn_print = Button(plt.axes([0.77, 0.34, 0.20, 0.04]), "Print spec")
    btn_save  = Button(plt.axes([0.77, 0.40, 0.20, 0.04]), "Save snapshot")

    # Reversed toggle
    btn_rev = Button(plt.axes([0.77, 0.10, 0.20, 0.04]), "Toggle reversed")

    state = {"mode": "diverging_palette", "reversed": False}

    def named_label() -> str:
        return NAMED_DIVERGING[int(sl_named.val)]

    def build_cmap() -> tuple:
        if state["mode"] == "diverging_palette":
            cmap = sns.diverging_palette(
                int(sl_h_neg.val), int(sl_h_pos.val),
                s=int(sl_s.val), l=int(sl_l.val),
                sep=int(sl_sep.val), as_cmap=True,
            )
            label = (
                f"[diverging_palette] h_neg={int(sl_h_neg.val)} h_pos={int(sl_h_pos.val)} "
                f"s={int(sl_s.val)} l={int(sl_l.val)} sep={int(sl_sep.val)}"
            )
            spec = (
                f"sns.diverging_palette({int(sl_h_neg.val)}, {int(sl_h_pos.val)}, "
                f"s={int(sl_s.val)}, l={int(sl_l.val)}, sep={int(sl_sep.val)}, as_cmap=True)"
            )
        elif state["mode"] == "cubehelix mirrored":
            cmap = cubehelix_mirrored(sl_start.val, sl_rot.val, sl_dark.val, sl_light.val)
            label = (
                f"[cubehelix mirrored] start={sl_start.val:.2f} rot={sl_rot.val:.2f} "
                f"dark={sl_dark.val:.2f} light={sl_light.val:.2f}"
            )
            spec = (
                f"# mirrored cubehelix — paste this helper from scripts/cmap_interactive.py\n"
                f"# cubehelix_mirrored(start={sl_start.val:.2f}, rot={sl_rot.val:.2f}, "
                f"dark={sl_dark.val:.2f}, light={sl_light.val:.2f})"
            )
        else:
            name = named_label()
            cmap = sns.color_palette(name, as_cmap=True)
            label = f"[named] {name}"
            spec = f'sns.color_palette("{name}", as_cmap=True)'

        if state["reversed"]:
            cmap = cmap.reversed()
            label += " (reversed)"
            spec += "  # .reversed()"
        return cmap, label, spec

    def update(_=None) -> None:
        cmap, label, _ = build_cmap()
        im.set_cmap(cmap)
        ax.set_title(f"RSA cmap viewer — {region}, Block {block}\n{label}", fontsize=11)
        fig.canvas.draw_idle()

    for sl in (sl_h_neg, sl_h_pos, sl_s, sl_l, sl_sep, sl_start, sl_rot, sl_dark, sl_light, sl_named):
        sl.on_changed(update)
    radio.on_clicked(lambda label: (state.update(mode=label), update()))

    def on_print(_):
        _, label, spec = build_cmap()
        print(f"\n--- selected ---\n{label}\nDIVERGING_CMAP = {spec}\n")

    def on_save(_):
        _, label, _ = build_cmap()
        out = ROOT / "outputs" / "pipeline" / "cmap_compare" / f"snapshot_{region}_b{block}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"-> saved: {out} ({label})")

    def on_toggle(_):
        state["reversed"] = not state["reversed"]
        update()

    btn_print.on_clicked(on_print)
    btn_save.on_clicked(on_save)
    btn_rev.on_clicked(on_toggle)

    plt.show()


if __name__ == "__main__":
    region_arg = sys.argv[1] if len(sys.argv) > 1 else "NYC"
    block_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(region=region_arg, block=block_arg)
