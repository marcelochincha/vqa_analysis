from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.style import DIVERGING_CMAP, apply_style, save_figure, styled_heatmap
from pipeline.utils.checkpoint import cached_dataframe
from pipeline.utils.io import load_csv, load_embeddings_cache
from pipeline.utils.metrics import (
    assign_block,
    display_agent_names,
    get_ordered_agents,
    get_video_sector,
    plot_group_mean_std_grid,
)


def run(config: PipelineConfig, show_progress: bool = False, force_recompute: bool = False) -> Path:
    apply_style()
    data_path = config.resolve(config.data_file)
    embeddings_path = config.resolve(config.embeddings_file)
    outdir = config.out_path("cosine")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(data_path, keep_default_na=False)
    df["VIDEO_SECTOR"] = df["VIDEO"].apply(get_video_sector)
    df["REPETITION"] = df["REPETITION"].astype(np.uint8)
    df["QUESTION_NUM"] = df["QUESTION_NUM"].astype(np.uint8)

    agent_order = get_ordered_agents(df["AGENT"].unique())

    def _compute_cosine() -> pd.DataFrame:
        cache = load_embeddings_cache(embeddings_path)
        df_first = df[df["REPETITION"] == 1].reset_index(drop=True)
        groups = list(df_first.groupby(["VIDEO", "QUESTION_NUM"]))
        iterator = tqdm(groups, desc="Computing cosine", unit="group") if show_progress else groups

        results = []
        for (video, qnum), group in iterator:
            embeddings_arr = []
            group_agent_order = []
            for row in group.itertuples(index=False):
                key = (row.AGENT, video, qnum, row.REPETITION)
                if key in cache:
                    embeddings_arr.append(cache[key])
                    group_agent_order.append(row.AGENT)

            if len(embeddings_arr) < 2:
                continue

            embeds = np.vstack(embeddings_arr)
            dists = cdist(embeds, embeds, metric="cosine")
            sims = 1 - dists

            sector = group["VIDEO_SECTOR"].iloc[0]
            for i, agent_i in enumerate(group_agent_order):
                for j, agent_j in enumerate(group_agent_order):
                    results.append(
                        {
                            "VIDEO": video,
                            "QUESTION_NUM": qnum,
                            "VIDEO_SECTOR": sector,
                            "AGENT_I": agent_i,
                            "AGENT_J": agent_j,
                            "RESULT": sims[i, j],
                        }
                    )

        agg = pd.DataFrame(results)
        if agg.empty:
            raise ValueError("No cosine similarity data computed")
        agg["BLOCK"] = agg["QUESTION_NUM"].apply(assign_block)
        return agg

    agg_df = cached_dataframe(
        outdir / "cosine_similarity_data.parquet",
        _compute_cosine,
        force=force_recompute,
        label="cosine",
    )
    agg_df2 = agg_df.groupby(["VIDEO_SECTOR", "AGENT_I", "AGENT_J", "BLOCK"], as_index=False)["RESULT"].mean()

    sectors = [s for s in ["Lima", "NYC"] if s in agg_df2["VIDEO_SECTOR"].unique()]
    blocks = sorted(agg_df2["BLOCK"].unique())
    nrows = len(sectors)
    ncols = len(blocks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    cmap = DIVERGING_CMAP
    for i, sector in enumerate(sectors):
        for j, block in enumerate(blocks):
            ax = axes[i, j]
            df_block = agg_df2[(agg_df2["VIDEO_SECTOR"] == sector) & (agg_df2["BLOCK"] == block)]
            if df_block.empty:
                ax.set_title(f"Block {block} - {sector}\n(Sin datos)")
                ax.axis("off")
                continue
            heatmap_data = df_block.pivot(index="AGENT_I", columns="AGENT_J", values="RESULT")
            heatmap_data = heatmap_data.reindex(index=agent_order, columns=agent_order)
            styled_heatmap(
                heatmap_data.to_numpy(),
                ax=ax,
                annot=False,
                cmap=cmap,
                xticklabels=display_agent_names(heatmap_data.columns),
                yticklabels=display_agent_names(heatmap_data.index),
                vmin=-1,
                vmax=1,
            )
            ax.set_title(f"Block {block} - {sector}", weight="bold", fontsize=16)

    embed_name = Path(embeddings_path).stem
    fig.suptitle(f"Cosine similarity heatmaps by block and region [{embed_name}]", fontsize=24, weight="bold")
    fig.tight_layout()
    out_path = outdir / "cosine_heatmap_grid.png"
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)

    plot_group_mean_std_grid(
        agg_df2,
        "RESULT",
        f"Cosine similarity - Group Mean±Std (Human Lima vs Human NYC vs VLM) [{embed_name}]",
        outdir / "cosine_group_meanstd_grid.png",
        vmin=-1,
        vmax=1,
    )
    return out_path