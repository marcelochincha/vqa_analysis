from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.style import apply_style
from pipeline.utils.io import load_csv, load_embeddings_cache


def categorize_agent(agent: str) -> str:
    agent = agent.lower()
    if "lima" in agent:
        return "Human_lima"
    elif "nyc" in agent:
        return "Human_nyc"
    return "Vlm"


def run(config: PipelineConfig, show_progress: bool = False) -> Path:
    apply_style()
    data_path = config.resolve(config.data_file)
    embeddings_path = config.resolve(config.embeddings_file)
    outdir = config.out_path("rsa")
    outdir.mkdir(parents=True, exist_ok=True)

    cache = load_embeddings_cache(embeddings_path)
    df = load_csv(data_path, keep_default_na=False)
    df["VIDEO_SECTOR"] = df["VIDEO"].apply(lambda x: "Lima" if int(x.split("_")[1]) <= 100 else "NYC")
    df_first = df[df["REPETITION"] == 1].reset_index(drop=True)

    embeddings = []
    for row in tqdm(df_first.itertuples(index=False), desc="Loading embeddings", unit="row", disable=not show_progress):
        key = (row.AGENT, row.VIDEO, row.QUESTION_NUM, row.REPETITION)
        if key not in cache:
            raise KeyError(f"Embedding for key {key} not found")
        embeddings.append(cache[key])
    embeddings_arr = np.vstack(embeddings)

    agents = list(df_first["AGENT"].unique())
    human_lima = sorted([a for a in agents if "human" in a.lower() and "lima" in a.lower()])
    human_nyc = sorted([a for a in agents if "human" in a.lower() and "nyc" in a.lower()])
    vlms = sorted([a for a in agents if "human" not in a.lower()])
    agent_order = vlms + human_lima + human_nyc

    gramians = {}
    rsa_rows = []
    sectors_list = list(df_first["VIDEO_SECTOR"].unique())
    blocks_list = sorted(df_first["BLOCK"].unique().astype(int))
    total_iterations = len(sectors_list) * len(blocks_list)

    pbar = tqdm(total=total_iterations, desc="Computing RSA", unit="sector/block", disable=not show_progress)
    for region in sectors_list:
        for block in blocks_list:
            for agent in agent_order:
                mask = (df_first["AGENT"] == agent) & (df_first["VIDEO_SECTOR"] == region) & (df_first["BLOCK"] == block)
                selected = df_first[mask]
                if not selected.empty:
                    embeds = embeddings_arr[selected.index]
                    embeds_norm = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
                    gramians[(agent, block, region)] = embeds_norm @ embeds_norm.T
            pbar.update(1)
    pbar.close()

    pbar = tqdm(total=len(agent_order) ** 2 * len(sectors_list) * len(blocks_list), desc="RSA correlations", unit="pair", disable=not show_progress)
    for region in sectors_list:
        for block in blocks_list:
            for i, agent_i in enumerate(agent_order):
                for j, agent_j in enumerate(agent_order):
                    gi = gramians.get((agent_i, block, region))
                    gj = gramians.get((agent_j, block, region))
                    if gi is None or gj is None:
                        continue
                    triu_idx = np.triu_indices_from(gi, k=1)
                    corr, _ = pearsonr(gi[triu_idx], gj[triu_idx])
                    rsa_rows.append(
                        {
                            "VIDEO_SECTOR": region,
                            "BLOCK": block,
                            "AGENT_I": agent_i,
                            "AGENT_J": agent_j,
                            "CORRELATION": corr,
                        }
                    )
                    pbar.update(1)
    pbar.close()

    rsa_df = pd.DataFrame(rsa_rows)
    if rsa_df.empty:
        raise ValueError("No RSA data computed")

    sectors = list(df_first["VIDEO_SECTOR"].unique())[::-1]
    blocks = sorted(df_first["BLOCK"].unique().astype(int))
    nrows, ncols = len(sectors), len(blocks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    for id_r, region in enumerate(sectors):
        for id_b, block in enumerate(blocks):
            ax = axes[id_r, id_b]
            df_block = rsa_df[(rsa_df["VIDEO_SECTOR"] == region) & (rsa_df["BLOCK"] == block)]
            if df_block.empty:
                ax.axis("off")
                continue
            rsa_matrix = df_block.pivot(index="AGENT_I", columns="AGENT_J", values="CORRELATION")
            rsa_matrix = rsa_matrix.reindex(index=agent_order, columns=agent_order)
            sns.heatmap(
                rsa_matrix.to_numpy(),
                annot=False,
                xticklabels=rsa_matrix.columns,
                yticklabels=rsa_matrix.index,
                cmap=cmap,
                ax=ax,
                square=True,
                vmin=-1,
                vmax=1,
            )
            ax.set_title(f"Region: {region}, Block: {block}", fontsize=16, weight="bold")

    fig.suptitle("RSA analysis - Representational Similarity Analysis", fontsize=24, weight="bold")
    fig.tight_layout()
    out_path = outdir / "rsa_heatmap_grid.png"
    fig.savefig(out_path, dpi=300) #, bbox_inches="tight")
    plt.close(fig)
    return out_path