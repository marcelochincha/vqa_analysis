"""Render the RSA noise-ceiling figure (Fig. supp-noiseceiling) from the JSON.

Reads rsa_noise_ceiling_<which>.json (produced by rsa_noise_ceiling.py) and writes:
  rebuttal/outputs/rsa_noise_ceiling.png
  rebuttal/graphics/rsa_noise_ceiling.pdf / .png

Two panels (Lima, NYC). Per block: grey band = human noise ceiling
[lower, upper]; blue dot = VLM-human RSA with bootstrap CI; orange dash =
human-human RSA. Labels are ASCII-only (no mathtext arrow) to avoid the
glyph-substitution bug that rendered "VLM->human" as "VLMohuman".
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_io import OUTDIR

BLOCKS = ["Factual", "Ratings", "Counterfactual", "Reasoning"]
GREY, BLUE, ORANGE = "#c8c8c8", "#2c7fb8", "#d95f02"


def _band(ax, x, lo, hi, w=0.5):
    ax.add_patch(plt.Rectangle((x - w / 2, lo), w, max(hi - lo, 1e-3),
                               facecolor=GREY, edgecolor="none", zorder=1))


def figure(which="mpnet"):
    rows = json.load(open(os.path.join(OUTDIR, f"rsa_noise_ceiling_{which}.json")))
    by = {(r["block_name"], r["region"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, region in zip(axes, ["Lima", "NYC"]):
        for x, blk in enumerate(BLOCKS):
            r = by[(blk, region)]
            _band(ax, x, r["nc_lower"], r["nc_upper"])
            ax.errorbar(x, r["vlm_to_human"],
                        yerr=[[r["vlm_to_human"] - r["vlm_to_human_lo"]],
                              [r["vlm_to_human_hi"] - r["vlm_to_human"]]],
                        fmt="o", color=BLUE, ms=8, capsize=4, zorder=3)
            ax.hlines(r["human_human"], x - 0.22, x + 0.22, color=ORANGE, lw=3, zorder=2)
        ax.set_xticks(range(len(BLOCKS)))
        ax.set_xticklabels(BLOCKS, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(region)
        ax.grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_ylabel("RSA (Pearson of RDMs)")

    legend = [
        Patch(facecolor=GREY, label="human noise ceiling"),
        Line2D([0], [0], marker="o", color=BLUE, ls="", ms=8, label="VLM-human"),
        Line2D([0], [0], color=ORANGE, lw=3, label="human-human"),
    ]
    axes[1].legend(handles=legend, loc="lower right", framealpha=0.9)
    fig.suptitle(f"RSA with human noise ceiling (all-{which})", fontweight="bold")
    fig.tight_layout()

    gdir = os.path.join(OUTDIR, "..", "graphics")
    os.makedirs(gdir, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "rsa_noise_ceiling.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(gdir, "rsa_noise_ceiling.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(gdir, "rsa_noise_ceiling.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote rsa_noise_ceiling.pdf/.png (label fixed: VLM-human)")


if __name__ == "__main__":
    import sys
    figure(sys.argv[1] if len(sys.argv) > 1 else "mpnet")
