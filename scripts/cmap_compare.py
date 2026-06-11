"""Render the RSA heatmap (one region/block) under many seaborn cmaps for visual comparison.

Output: outputs/pipeline/cmap_compare/rsa_cmap_grid.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.utils.metrics import get_ordered_agents


# (label_for_title, cmap_object_or_name). Big sweep across seaborn + matplotlib cmaps.
CMAPS = [
    # Seaborn perceptual sequential
    ("rocket", "rocket"),
    ("rocket_r", "rocket_r"),
    ("mako", "mako"),
    ("mako_r", "mako_r"),
    ("flare", "flare"),
    ("flare_r", "flare_r"),
    ("crest", "crest"),
    ("crest_r", "crest_r"),
    # Matplotlib perceptual sequential
    ("viridis", "viridis"),
    ("viridis_r", "viridis_r"),
    ("magma", "magma"),
    ("magma_r", "magma_r"),
    ("inferno", "inferno"),
    ("plasma", "plasma"),
    ("cividis", "cividis"),
    ("turbo", "turbo"),
    # Cubehelix variants
    ("cubehelix default", sns.cubehelix_palette(as_cmap=True)),
    ("cubehelix s=2,r=0", sns.cubehelix_palette(start=2, rot=0, dark=0.2, light=0.95, as_cmap=True)),
    ("cubehelix s=.5,r=-.5", sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)),
    ("cubehelix s=3,r=.4", sns.cubehelix_palette(start=3, rot=0.4, dark=0.15, light=0.9, as_cmap=True)),
    # Single-hue sequential
    ("Blues", "Blues"),
    ("Reds", "Reds"),
    ("Greens", "Greens"),
    ("Purples", "Purples"),
    ("Oranges", "Oranges"),
    ("Greys", "Greys"),
    # Multi-hue sequential
    ("YlOrRd", "YlOrRd"),
    ("YlGnBu", "YlGnBu"),
    ("BuPu", "BuPu"),
    ("PuBuGn", "PuBuGn"),
    # Diverging (seaborn)
    ("vlag (div)", "vlag"),
    ("icefire (div)", "icefire"),
    # Diverging (matplotlib)
    ("coolwarm (div)", "coolwarm"),
    ("RdBu_r (div)", "RdBu_r"),
    ("RdYlBu_r (div)", "RdYlBu_r"),
    ("Spectral_r (div)", "Spectral_r"),
    ("PiYG (div)", "PiYG"),
    ("PRGn (div)", "PRGn"),
    ("BrBG (div)", "BrBG"),
    ("PuOr (div)", "PuOr"),
    ("seismic (div)", "seismic"),
    ("bwr (div)", "bwr"),
]


def main(region: str = "NYC", block: int = 1) -> Path:
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
    rsa_matrix = (
        df_block.pivot(index="AGENT_I", columns="AGENT_J", values="CORRELATION")
        .reindex(index=agent_order, columns=agent_order)
    )

    ncols = 6
    nrows = (len(CMAPS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes_flat = axes.flatten()

    for i, (label, cmap) in enumerate(CMAPS):
        ax = axes_flat[i]
        sns.heatmap(
            rsa_matrix.to_numpy(),
            ax=ax,
            cmap=cmap,
            square=True,
            vmin=-1,
            vmax=1,
            xticklabels=False,
            yticklabels=False,
            cbar=True,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(label, fontsize=11, weight="bold")

    for j in range(len(CMAPS), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Seaborn cmap comparison — RSA ({region}, Block {block}), vmin=-1, vmax=1",
        fontsize=18,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = ROOT / "outputs" / "pipeline" / "cmap_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rsa_cmap_grid_{region}_b{block}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")
    return out_path


if __name__ == "__main__":
    region = sys.argv[1] if len(sys.argv) > 1 else "NYC"
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(region=region, block=block)
