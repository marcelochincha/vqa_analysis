from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pipeline.config import PipelineConfig
from pipeline.style import COLORS, apply_style
from pipeline.utils.checkpoint import cached_dataframe
from pipeline.utils.io import load_csv, load_embeddings_cache


def get_group_color(agent: str) -> str:
    agent = agent.lower()
    if "human" in agent:
        return "nyc" if "nyc" in agent else "lima"
    return "vlm"


def run(config: PipelineConfig, force_recompute: bool = False) -> Path:
    apply_style()
    data_path = config.resolve(config.data_file)
    embeddings_path = config.resolve(config.embeddings_file)
    outdir = config.out_path("embed")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(data_path, keep_default_na=False)
    df["VIDEO_SECTOR"] = df["VIDEO"].apply(lambda x: "Lima" if int(x.split("_")[1]) <= 100 else "NYC")
    df_first = df[df["REPETITION"] == 1].reset_index(drop=True)

    def _compute_pca() -> pd.DataFrame:
        cache = load_embeddings_cache(embeddings_path)
        embeddings = []
        for row in df_first.itertuples(index=False):
            key = (row.AGENT, row.VIDEO, row.QUESTION_NUM, row.REPETITION)
            if key not in cache:
                raise KeyError(f"Embedding for key {key} not found")
            embeddings.append(cache[key])
        embeddings_arr = np.vstack(embeddings)

        coords = np.zeros((len(df_first), 2))
        for block in sorted(df_first["BLOCK"].unique()):
            mask = df_first["BLOCK"] == block
            if mask.any():
                coords[mask.values] = PCA(n_components=2).fit_transform(embeddings_arr[mask.values])

        out = df_first.assign(pca_X=coords[:, 0], pca_Y=coords[:, 1])
        out["group"] = out["AGENT"].map(get_group_color)
        return out

    df_plot = cached_dataframe(
        outdir / "pca_coords.parquet",
        _compute_pca,
        force=force_recompute,
        label="embed",
    )

    sectors = ["Lima", "NYC"]
    blocks = sorted(df_plot["BLOCK"].unique())
    fig, axes = plt.subplots(len(sectors), len(blocks), figsize=(6 * len(blocks), 5 * len(sectors)), sharex=True, sharey=True)
    if len(sectors) == 1 or len(blocks) == 1:
        axes = np.array(axes).reshape(len(sectors), len(blocks))

    min_x, max_x = df_plot["pca_X"].min(), df_plot["pca_X"].max()
    min_y, max_y = df_plot["pca_Y"].min(), df_plot["pca_Y"].max()
    x_pad = (max_x - min_x) * 0.05
    y_pad = (max_y - min_y) * 0.05

    handles_dict = {}
    for i, sector in enumerate(sectors):
        for j, block in enumerate(blocks):
            ax = axes[i, j]
            df_subset = df_plot[(df_plot["BLOCK"] == block) & (df_plot["VIDEO_SECTOR"] == sector)]
            if df_subset.empty:
                ax.set_title(f"Block {block} - {sector}\n(Sin datos)")
                ax.axis("off")
                continue
            for agent in df_subset["AGENT"].unique():
                mask = df_subset["AGENT"] == agent
                group = df_subset.loc[mask, "group"].iloc[0]
                sc = ax.scatter(
                    df_subset.loc[mask, "pca_X"],
                    df_subset.loc[mask, "pca_Y"],
                    label=agent,
                    color=COLORS[group],
                    alpha=0.6,
                    edgecolors="white",
                    s=30,
                    linewidth=0.5,
                )
                handles_dict[agent] = sc
            ax.set_title(f"Block {block} - {sector}", weight="bold", fontsize=14)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.set_xlim(min_x - x_pad, max_x + x_pad)
            ax.set_ylim(min_y - y_pad, max_y + y_pad)
            ax.grid(True, alpha=0.3)

    fig.legend(
        handles=list(handles_dict.values()),
        labels=list(handles_dict.keys()),
        loc="center right",
        frameon=True,
        bbox_to_anchor=(1.03, 0.5),
        fontsize=12,
        title_fontproperties={"weight": "bold", "size": 14},
        title="Agent",
    )
    fig.suptitle("PCA by Block and Sector - Embeddings", fontsize=24, weight="bold")
    out_path = outdir / "pca_by_block_sector.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path