from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=float).ravel()
    b = np.asarray(vec_b, dtype=float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(a, b) / denom)


def get_row_key(row) -> tuple:
    return (row["AGENT"], row["VIDEO"], row["QUESTION_NUM"], row["REPETITION"])


def build_row_map(df: pd.DataFrame, embeddings_cache: dict) -> dict:
    row_map = {}
    for row in df.itertuples(index=False):
        key = (row.AGENT, row.VIDEO, row.QUESTION_NUM, row.REPETITION)
        if key in embeddings_cache:
            row_map[key] = embeddings_cache[key]
        elif row.ANSWER in embeddings_cache:
            row_map[key] = embeddings_cache[row.ANSWER]
        else:
            raise KeyError(f"Embedding for key {key} (or answer text) not found in cache")
    return row_map


def assign_block(question_num: int) -> int:
    if question_num <= 5:
        return 1
    if question_num <= 10:
        return 2
    if question_num <= 15:
        return 3
    return 4


def _natural_key(name: str) -> list:
    import re
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


_NYC_RENUMBER = None


def display_agent_name(agent: str) -> str:
    # human_nyc_11..20 → human_nyc_1..10 so NYC numbering mirrors Lima.
    # Display-only — internal identifiers (cache keys, parquet rows) keep the original ID.
    import re
    global _NYC_RENUMBER
    if _NYC_RENUMBER is None:
        _NYC_RENUMBER = re.compile(r"^(human_nyc_)(\d+)$", re.IGNORECASE)
    m = _NYC_RENUMBER.match(str(agent))
    if m and 11 <= int(m.group(2)) <= 20:
        return f"{m.group(1)}{int(m.group(2)) - 10}"
    return str(agent)


def display_agent_names(agents) -> list[str]:
    return [display_agent_name(a) for a in agents]


def get_ordered_agents(agents) -> list[str]:
    agents = list(dict.fromkeys(agents))
    human_lima = sorted((a for a in agents if "human" in a.lower() and "lima" in a.lower()), key=_natural_key)
    human_nyc = sorted((a for a in agents if "human" in a.lower() and "nyc" in a.lower()), key=_natural_key)
    vlms = sorted((a for a in agents if "human" not in a.lower()), key=_natural_key)
    return vlms + human_lima + human_nyc


def categorize_agent_group(agent: str) -> str:
    """Collapse an agent id into one of the three analysis groups."""
    a = str(agent).lower()
    if "human" in a and "lima" in a:
        return "Human_lima"
    if "human" in a and "nyc" in a:
        return "Human_nyc"
    return "Vlm"


def plot_group_mean_std_grid(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path,
    *,
    vmin: float,
    vmax: float,
    center: float = 0.0,
    mirror: bool = False,
    sector_col: str = "VIDEO_SECTOR",
    block_col: str = "BLOCK",
):
    """Summary 3x3 group-level mean±std heatmaps, in a sector x block grid.

    Collapses agents into Human_lima / Human_nyc / Vlm and, for each
    (sector, block), draws a 3x3 matrix coloured by the group-pair mean of
    ``value_col`` and annotated as ``mean±std`` across the agent pairs in that
    cell. Set ``mirror=True`` when the input stores only one direction per
    agent pair (e.g. deduped judge pairs) so off-diagonal cells aggregate both
    triangles; leave it False when both (i, j) and (j, i) rows already exist.
    """
    import matplotlib.pyplot as plt

    from pipeline.style import DIVERGING_CMAP, save_figure, styled_heatmap

    df = df.copy()
    df["GROUP_I"] = df["AGENT_I"].apply(categorize_agent_group)
    df["GROUP_J"] = df["AGENT_J"].apply(categorize_agent_group)
    if mirror:
        df = pd.concat(
            [df, df.rename(columns={"GROUP_I": "GROUP_J", "GROUP_J": "GROUP_I"})],
            ignore_index=True,
        )

    group_order = ["Human_lima", "Human_nyc", "Vlm"]
    sectors = list(df[sector_col].unique())[::-1]
    blocks = sorted(df[block_col].astype(int).unique())
    nrows, ncols = len(sectors), len(blocks)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True, squeeze=False
    )

    for i, sector in enumerate(sectors):
        for j, block in enumerate(blocks):
            ax = axes[i, j]
            sub = df[(df[sector_col] == sector) & (df[block_col].astype(int) == int(block))]
            mean_mtx = sub.pivot_table(
                index="GROUP_I", columns="GROUP_J", values=value_col, aggfunc="mean"
            ).reindex(index=group_order, columns=group_order)
            std_mtx = sub.pivot_table(
                index="GROUP_I", columns="GROUP_J", values=value_col, aggfunc="std"
            ).reindex(index=group_order, columns=group_order)

            annot = []
            for r in range(len(group_order)):
                row = []
                for c in range(len(group_order)):
                    m = mean_mtx.iloc[r, c]
                    s = std_mtx.iloc[r, c]
                    if pd.isna(m):
                        row.append("-")
                    elif pd.isna(s):
                        row.append(f"{m:.2f}±0.00")
                    else:
                        row.append(f"{m:.2f}±{s:.2f}")
                annot.append(row)

            styled_heatmap(
                mean_mtx.to_numpy(),
                ax=ax,
                annot=annot,
                fmt="",
                cmap=DIVERGING_CMAP,
                center=center,
                vmin=vmin,
                vmax=vmax,
                xticklabels=group_order,
                yticklabels=group_order,
                annot_kws={"fontsize": 10, "weight": "bold"},
            )
            ax.set_title(f"Region: {sector}, Block: {block}", fontsize=14, weight="bold")

    fig.suptitle(title, fontsize=20, weight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def get_video_sector(video_str: str) -> str:
    try:
        video_num = int(video_str.split("_")[1])
        return "Lima" if video_num <= 100 else "NYC"
    except (IndexError, ValueError):
        return "Unknown"


def get_video_region(video_str: str) -> str:
    try:
        video_num = int(video_str.split("_")[1])
        return "Videos of LIMA" if video_num <= 100 else "Videos of NYC"
    except (IndexError, ValueError):
        return "Unknown"


def to_numeric(answer) -> float | np.nan:
    try:
        return float(answer)
    except Exception:
        return np.nan


def compute_region_stats(df_vlms: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for agent in df_vlms["AGENT"].unique():
        df_agent = df_vlms[df_vlms["AGENT"] == agent]
        for question in df_vlms["QUESTION_NUM"].unique():
            df_values = df_agent[df_agent["QUESTION_NUM"] == question][["VIDEO_REGION", "ANSWER"]]
            region_values = [df_values[df_values["VIDEO_REGION"] == region]["ANSWER"].tolist() for region in df_values["VIDEO_REGION"].unique()]
            if len(region_values) != 2:
                continue
            try:
                stat, pvalue = ks_2samp(region_values[0], region_values[1])
            except Exception:
                stat, pvalue = np.nan, np.nan
            stats.append(
                {
                    "AGENT": agent,
                    "QUESTION_NUM": question,
                    "WASSERSTEIN_DISTANCE": wasserstein_distance(region_values[0], region_values[1]),
                    "KS_2SAMP_STATISTIC": stat,
                    "KS_2SAMP_PVALUE": pvalue,
                }
            )
    return pd.DataFrame(stats)