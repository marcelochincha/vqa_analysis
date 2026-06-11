from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pipeline.config import PipelineConfig
from pipeline.style import COLORS, apply_style, build_agent_markers, save_figure
from pipeline.utils.checkpoint import cached_dataframe
from pipeline.utils.io import load_csv, load_embeddings_cache
from pipeline.utils.metrics import display_agent_name, get_ordered_agents

# Fixed seed shared by every stochastic reducer so runs are reproducible.
RANDOM_STATE = 42

# Each method: how to project a (n_samples, n_features) block to 2D, plus axis labels.
# t-SNE / UMAP are made deterministic via a fixed seed + deterministic init + single thread.
METHODS = {
    "pca": {"title": "PCA", "axis": ("PC 1", "PC 2")},
    "tsne": {"title": "t-SNE", "axis": ("t-SNE 1", "t-SNE 2")},
    "umap": {"title": "UMAP", "axis": ("UMAP 1", "UMAP 2")},
}


def get_group_color(agent: str) -> str:
    agent = agent.lower()
    if "human" in agent:
        return "nyc" if "nyc" in agent else "lima"
    return "vlm"


def _reduce(embeddings: np.ndarray, method: str) -> np.ndarray:
    """Project embeddings to 2D with a deterministic, reproducible reducer.

    Fitted ONCE on the full dataset (all blocks/sectors) so every point lives in
    a single, comparable 2D space; the plotting layer only filters which subset
    to show. Fitting per-block instead would isolate the ordinal Block 2 (~10
    distinct embeddings) and make t-SNE/UMAP degenerate.
    """
    n = len(embeddings)
    if n < 3:
        # Too few points for a meaningful neighbour-based embedding; pad to 2D safely.
        coords = np.zeros((n, 2))
        coords[:, : min(2, embeddings.shape[1])] = embeddings[:, : min(2, embeddings.shape[1])]
        return coords

    if method == "pca":
        return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings)

    if method == "tsne":
        # init="pca" + fixed seed => deterministic. n_jobs=1 avoids thread-order nondeterminism.
        perplexity = 800
        print(f"Running t-SNE with perplexity={perplexity:.1f} on {n} samples")
        tsne = TSNE(
            n_components=2,
            random_state=RANDOM_STATE,
            init="pca",
            perplexity=perplexity,
            n_jobs=1,
            max_iter=1000,
        )
        return tsne.fit_transform(embeddings)

    if method == "umap":
        # Imported lazily: heavy dependency only needed when UMAP is requested.
        import umap

        # A set random_state forces UMAP onto a single thread internally => reproducible.
        n_neighbors = 50
        print(f"Running UMAP with n_neighbors={n_neighbors} on {n} samples")
        reducer = umap.UMAP(
            n_components=2,
            random_state=RANDOM_STATE,
            n_neighbors=n_neighbors,
            init="spectral",
            n_jobs=1,
            transform_seed=RANDOM_STATE,
        )
        return reducer.fit_transform(embeddings)

    raise ValueError(f"Unknown reduction method: {method}")


def _compute_coords(df_first: pd.DataFrame, embeddings_path: Path, method: str, scope: str) -> pd.DataFrame:
    cache = load_embeddings_cache(embeddings_path)
    embeddings = []
    for row in df_first.itertuples(index=False):
        key = (row.AGENT, row.VIDEO, row.QUESTION_NUM, row.REPETITION)
        if key not in cache:
            raise KeyError(f"Embedding for key {key} not found")
        embeddings.append(cache[key])
    embeddings_arr = np.vstack(embeddings)

    coords = np.zeros((len(df_first), 2))
    if scope == "global":
        # Fit once on ALL points -> single comparable 2D space; filter at plot time.
        coords = _reduce(embeddings_arr, method)
    elif scope == "perblock":
        # Fit each block independently -> each block gets its own optimal layout
        # (coordinates are NOT comparable across blocks).
        for block in sorted(df_first["BLOCK"].unique()):
            mask = (df_first["BLOCK"] == block).to_numpy()
            if mask.any():
                coords[mask] = _reduce(embeddings_arr[mask], method)
    else:
        raise ValueError(f"Unknown fit scope: {scope}")

    out = df_first.assign(dim_X=coords[:, 0], dim_Y=coords[:, 1])
    out["group"] = out["AGENT"].map(get_group_color)
    return out


def _axis_limits(df: pd.DataFrame) -> tuple[float, float, float, float]:
    min_x, max_x = df["dim_X"].min(), df["dim_X"].max()
    min_y, max_y = df["dim_Y"].min(), df["dim_Y"].max()
    x_pad = (max_x - min_x) * 0.05
    y_pad = (max_y - min_y) * 0.05
    return min_x - x_pad, max_x + x_pad, min_y - y_pad, max_y + y_pad


def _plot(df_plot: pd.DataFrame, method: str, scope: str, out_path: Path, embed_name: str = "") -> None:
    title = METHODS[method]["title"]
    xlabel, ylabel = METHODS[method]["axis"]

    sectors = ["Lima", "NYC"]
    blocks = sorted(df_plot["BLOCK"].unique())
    # Global fit: one comparable space -> share axes across the whole grid.
    # Per-block fit: each block lives in its own space -> limits set per column.
    share = scope == "global"
    fig, axes = plt.subplots(
        len(sectors), len(blocks), figsize=(6 * len(blocks), 5 * len(sectors)), sharex=share, sharey=share
    )
    if len(sectors) == 1 or len(blocks) == 1:
        axes = np.array(axes).reshape(len(sectors), len(blocks))

    if scope == "global":
        global_limits = _axis_limits(df_plot)
        block_limits = {block: global_limits for block in blocks}
    else:
        block_limits = {block: _axis_limits(df_plot[df_plot["BLOCK"] == block]) for block in blocks}

    legend_order = get_ordered_agents(df_plot["AGENT"].unique())
    agent_markers = build_agent_markers(legend_order)
    # Draw humans first so VLM points land ON TOP — independent of the legend order above.
    plot_order = (
        [a for a in legend_order if "human" in a.lower()]
        + [a for a in legend_order if "human" not in a.lower()]
    )
    handles_dict = {}
    for i, sector in enumerate(sectors):
        for j, block in enumerate(blocks):
            ax = axes[i, j]
            df_subset = df_plot[(df_plot["BLOCK"] == block) & (df_plot["VIDEO_SECTOR"] == sector)]
            if df_subset.empty:
                ax.set_title(f"Block {block} - {sector}\n(Sin datos)")
                ax.axis("off")
                continue
            for agent in plot_order:
                mask = df_subset["AGENT"] == agent
                if not mask.any():
                    continue
                group = df_subset.loc[mask, "group"].iloc[0]
                sc = ax.scatter(
                    df_subset.loc[mask, "dim_X"],
                    df_subset.loc[mask, "dim_Y"],
                    label=agent,
                    color=COLORS[group],
                    marker=agent_markers[agent],
                    alpha=0.6,
                    edgecolors="white",
                    s=40,
                    linewidth=0.5,
                )
                handles_dict.setdefault(agent, sc)
            ax.set_title(f"Block {block} - {sector}", weight="bold", fontsize=14)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            xmin, xmax, ymin, ymax = block_limits[block]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)

    ordered_handles = [handles_dict[a] for a in legend_order if a in handles_dict]
    ordered_labels = [display_agent_name(a) for a in legend_order if a in handles_dict]
    fig.legend(
        handles=ordered_handles,
        labels=ordered_labels,
        loc="center right",
        frameon=True,
        bbox_to_anchor=(1.03, 0.5),
        fontsize=12,
        title_fontproperties={"weight": "bold", "size": 14},
        title="Agent",
    )
    scope_label = "global fit" if scope == "global" else "per-block fit"
    embed_suffix = f" [{embed_name}]" if embed_name else ""
    fig.suptitle(f"{title} by Block and Sector - Embeddings ({scope_label}){embed_suffix}", fontsize=24, weight="bold")
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(config: PipelineConfig, force_recompute: bool = False) -> Path:
    apply_style()
    np.random.seed(RANDOM_STATE)
    data_path = config.resolve(config.data_file)
    embeddings_path = config.resolve(config.embeddings_file)
    outdir = config.out_path("embed")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(data_path, keep_default_na=False)
    df["VIDEO_SECTOR"] = df["VIDEO"].apply(lambda x: "Lima" if int(x.split("_")[1]) <= 100 else "NYC")
    df_first = df[df["REPETITION"] == 1].reset_index(drop=True)

    out_paths = {}
    for method in METHODS:
        for scope in ("global", "perblock"):
            df_plot = cached_dataframe(
                outdir / f"{method}_{scope}_coords.parquet",
                lambda method=method, scope=scope: _compute_coords(df_first, embeddings_path, method, scope),
                force=force_recompute,
                label=f"embed:{method}:{scope}",
            )
            out_path = outdir / f"{method}_{scope}_by_block_sector.png"
            embed_name = Path(embeddings_path).stem
            _plot(df_plot, method, scope, out_path, embed_name=embed_name)
            out_paths[(method, scope)] = out_path

    return out_paths[("pca", "global")]
