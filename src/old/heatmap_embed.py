"""Heatmap analysis for cosine similarity of embeddings."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from src.heatmap_common import reorder_matrix_by_groups
from src.old import config
import argparse
from itertools import combinations,combinations_with_replacement
from tqdm import tqdm
from sklearn.decomposition import PCA

from src.old import utils

def check_embedding_cache(cache_path):
    """Check if the embedding cache exists (keyed or legacy)."""
    keyed_path = config.EMBEDDING_CACHE["keyed"]
    legacy_path = config.EMBEDDING_CACHE["legacy_npy"]
    if not os.path.exists(keyed_path) and not os.path.exists(legacy_path):
        print("\n⚠ Embedding cache not found! Please run `embed_analysis.py` to generate embeddings first.\n")
        sys.exit(1)

def generate_score_similarity_heatmaps(metadata):
    """Generate combined 2x4 heatmap grids for cosine score mean/std (rows=region, cols=block)."""
    output_dir = config.OUTPUT_EMBEDDINGS_DIR
    utils.ensure_output_dir(output_dir)
    region_labels = {"lima": "Lima", "nyc": "NYC"}
    regions = ["lima", "nyc"]

    # Generate color map
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cmap", ["#ffffff", "#ee932c"])

    metadata = metadata.copy()
    if "VIDEO_REGION" not in metadata.columns:
        metadata["VIDEO_REGION"] = metadata["VIDEO"].apply(utils.infer_video_region)

    def _make_symmetric(matrix: pd.DataFrame, diagonal_value: float) -> pd.DataFrame:
        """Ensure matrix is symmetric and complete over all seen agents."""
        if matrix is None or matrix.empty:
            return pd.DataFrame()
        agents = sorted(set(matrix.index) | set(matrix.columns))
        matrix = matrix.reindex(index=agents, columns=agents)
        matrix = matrix.combine_first(matrix.T)
        matrix = matrix.fillna(matrix.T)
        matrix_values = matrix.to_numpy(copy=True)
        np.fill_diagonal(matrix_values, diagonal_value)
        matrix.iloc[:, :] = matrix_values
        return matrix

    def _save_grid(
        matrix_by_region_block,
        title: str,
        color_label: str,
        filename_base: str,
        cmap_name,
        vmin: float,
        vmax: float,
    ):
        """Save a 2x4 matrix of heatmaps with shared style and right-side colorbar."""
        import matplotlib.patches as patches
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        plt.style.use("default")
        valid_matrices = [m for m in matrix_by_region_block.values() if m is not None and not m.empty]
        max_agents = max((m.shape[0] for m in valid_matrices), default=10)
        subplot_size = max(6.8, min(10.5, max_agents * 0.44))
        fig_width = subplot_size * 4 + 1.2
        fig_height = subplot_size * 2 + 0.8

        annot_fontsize = 8 if max_agents > 22 else 9
        tick_fontsize = 12
        subtitle_fontsize = 32

        fig, axes = plt.subplots(2, 4, figsize=(fig_width, fig_height), constrained_layout=False)
        fig.suptitle(title, fontsize=48, fontweight="bold", y=0.985)
        fig.subplots_adjust(left=0.02, right=0.94, top=0.89, bottom=0.08, wspace=0.06, hspace=0.16)

        norm = Normalize(vmin=vmin, vmax=vmax)
        shared_mappable = ScalarMappable(norm=norm, cmap=cmap_name)

        for row_idx, region in enumerate(regions):
            for col_idx, block in enumerate(display_blocks):
                ax = axes[row_idx, col_idx]

                if block is None:
                    ax.axis("off")
                    continue

                matrix = matrix_by_region_block.get((region, block))
                if matrix is None or matrix.empty:
                    ax.set_axis_off()
                    ax.set_title(f"{region_labels[region]} | Block {block}\nNo data", fontsize=16)
                    continue

                matrix, group_sizes = reorder_matrix_by_groups(matrix)
                values = matrix.values
                agents = matrix.index.tolist()
                n_agents = len(agents)

                im = ax.pcolormesh(
                    np.arange(n_agents + 1),
                    np.arange(n_agents + 1),
                    values,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    zorder=1,
                    shading="flat",
                )

                for i in range(n_agents):
                    for j in range(n_agents):
                        val = values[i, j]
                        text = format(val, ".2f")
                        cell_color = im.cmap(im.norm(val))
                        brightness = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                        text_color = "black" if brightness > 0.5 else "white"
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            text,
                            ha="center",
                            va="center",
                            fontsize=annot_fontsize,
                            color=text_color,
                        )

                vlm_size, lima_size, nyc_size = group_sizes
                sizes = [vlm_size, lima_size, nyc_size]
                colors = [
                    config.PLOT_COLORS["VLM"],
                    config.PLOT_COLORS["HUMAN_LIMA"],
                    config.PLOT_COLORS["HUMAN_NYC"],
                ]

                start = 0
                for gsize, gcolor in zip(sizes, colors):
                    if gsize > 0:
                        rect = patches.Rectangle(
                            (start, start),
                            gsize,
                            gsize,
                            edgecolor=gcolor,
                            linewidth=2.0,
                            fill=False,
                            zorder=10,
                            antialiased=False,
                            joinstyle="miter",
                        )
                        ax.add_patch(rect)
                    start += gsize

                ax.set_xlim(0, n_agents)
                ax.set_ylim(n_agents, 0)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks(np.arange(n_agents) + 0.5)
                ax.set_yticks(np.arange(n_agents) + 0.5)

                ax.set_title(f"Block {block} - {region_labels[region]}", fontsize=subtitle_fontsize, fontweight="bold", pad=8)
                if row_idx == 1:
                    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=tick_fontsize)
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")

                if col_idx == 0:
                    ax.set_yticklabels(agents, fontsize=tick_fontsize)
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

                ax.tick_params(length=0)

        cbar = fig.colorbar(
            shared_mappable,
            ax=axes.ravel().tolist(),
            location="right",
            fraction=0.020,
            pad=0.008,
            shrink=0.97,
        )
        cbar.set_label(color_label, fontsize=16, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)
        for ext in ["png", "svg"]:
            out_path = os.path.join(output_dir, f"{filename_base}.{ext}")
            fig.savefig(out_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
            print(f"✓ Saved: {os.path.basename(out_path)}")

        plt.close(fig)

    unique_blocks = sorted(metadata["BLOCK"].dropna().unique())
    if not unique_blocks:
        print("⚠ No BLOCK values found. Skipping score similarity heatmaps.")
        return

    if len(unique_blocks) > 4:
        print(f"⚠ Found {len(unique_blocks)} blocks; using first 4 for 2x4 layout: {unique_blocks[:4]}")
    blocks = unique_blocks[:4]
    display_blocks = blocks + [None] * (4 - len(blocks))

    mean_matrices = {}
    std_matrices = {}

    for region in regions:
        region_metadata = metadata[metadata["VIDEO_REGION"] == region]
        for block in blocks:
            block_metadata = region_metadata[region_metadata["BLOCK"] == block]
            if len(block_metadata) < 1:
                continue

            df_avg = (
                block_metadata
                .groupby(["AGENT_I", "AGENT_J"], as_index=False)
                .SCORE.mean()
            )

            df_std = (
                block_metadata
                .groupby(["AGENT_I", "AGENT_J"], as_index=False)
                .SCORE.std()
                .fillna(0.0)
            )

            matrix = df_avg.pivot(index="AGENT_I", columns="AGENT_J", values="SCORE")
            matrix_std = df_std.pivot(index="AGENT_I", columns="AGENT_J", values="SCORE")

            mean_matrices[(region, block)] = _make_symmetric(matrix, diagonal_value=1.0)
            std_matrices[(region, block)] = _make_symmetric(matrix_std, diagonal_value=0.0)

    _save_grid(
        matrix_by_region_block=mean_matrices,
        title="Cosine Score Similarity (Mean) by Region and Block",
        color_label="Cosine Similarity",
        filename_base="cosine_score_similarity_grid_2x4",
        cmap_name=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    _save_grid(
        matrix_by_region_block=std_matrices,
        title="Cosine Score Variability (Std) by Region and Block",
        color_label="Cosine Similarity Std Dev",
        filename_base="cosine_score_std_grid_2x4",
        cmap_name="Reds",
        vmin=0.0,
        vmax=0.5,
    )

# def generate_cosine_agreement_heatmaps(pairwise_df: pd.DataFrame, metadata: pd.DataFrame):
#     """Generate agreement heatmaps for cosine similarity by block."""
#     output_dir = config.OUTPUT_EMBEDDINGS_DIR
#     utils.ensure_output_dir(output_dir)
#     region_labels = {"both": "All", "lima": "Lima", "nyc": "NYC"}

#     for block in sorted(metadata["BLOCK"].unique()):
#         for region in ["both", "lima", "nyc"]:
#             try:
#                 agreement_matrix = create_agreement_matrix(
#                     pairwise_df,
#                     block,
#                     score_column="COSINE_SCORE",
#                     threshold=0.5,
#                     video_region=region,
#                 )
#             except ValueError as exc:
#                 print(f"⚠ Skipping cosine agreement for block={block}, region={region}: {exc}")
#                 continue

#             if region == "both":
#                 agreement_path = os.path.join(output_dir, f"cosine_agreement_block{block}.png")
#             else:
#                 agreement_path = os.path.join(output_dir, f"cosine_agreement_block{block}_{region}.png")

#             region_title = "" if region == "both" else f" ({region_labels[region]})"
#             plot_agreement_heatmap(
#                 agreement_matrix,
#                 title=f"Cosine Agreement - Block {block}{region_title}",
#                 output_path=agreement_path,
#                 cmap="RdYlGn",
#                 use_group_colors=True
#             )

def process_vlm_embeddings(embeddings: np.ndarray, metadata: pd.DataFrame, mode: str = "mean", max_workers: int = 4) -> tuple:
    """Process VLM embeddings to handle multiple responses per question.

    Args:
        embeddings: Numpy array of all embeddings.
        metadata: DataFrame containing metadata for embeddings.
        mode: How to process VLM embeddings - "mean" (average) or "first" (use first response).

    Returns:
        Processed embeddings and metadata.
    """
    vlm_mask = metadata["AGENT"].isin(config.VLM_AGENTS)
    human_mask = ~vlm_mask
    
    # Fallback: if no VLM agents matched (dynamic discovery), detect by prefix
    if not vlm_mask.any():
        human_mask = metadata["AGENT"].str.startswith("human_")
        vlm_mask = ~human_mask

    # Separate humans and VLMs
    human_embeddings = embeddings[human_mask]
    human_metadata = metadata[human_mask]

    vlm_embeddings = embeddings[vlm_mask]
    vlm_metadata = metadata[vlm_mask].reset_index(drop=True)  # Reset indices to align with embeddings

    if mode == "first":
        # Keep only the first response for each (VIDEO, QUESTION_NUM, AGENT)
        vlm_metadata = vlm_metadata.drop_duplicates(subset=["VIDEO", "QUESTION_NUM", "AGENT"], keep="first")
        first_indices = vlm_metadata.index
        vlm_embeddings = vlm_embeddings[first_indices]
    elif mode == "mean":
        # Group by (VIDEO, QUESTION_NUM, AGENT) and average embeddings
        grouped = vlm_metadata.groupby(["VIDEO", "QUESTION_NUM", "AGENT"])
        mean_embeddings = []
        mean_metadata = []

        for _, group in grouped:
            indices = group.index
            mean_embeddings.append(vlm_embeddings[indices].mean(axis=0))
            mean_metadata.append(group.iloc[0])

        vlm_embeddings = np.vstack(mean_embeddings)
        vlm_metadata = pd.DataFrame(mean_metadata)

    # Combine humans and processed VLMs
    combined_embeddings = np.vstack([human_embeddings, vlm_embeddings])
    combined_metadata = pd.concat([human_metadata, vlm_metadata], ignore_index=True)

    print(f"Processed VLM embeddings using mode '{mode}': {len(vlm_metadata)} unique (VIDEO, QUESTION_NUM, AGENT) entries")
    print(f"Combined dataset: {len(combined_metadata)} total entries (Humans: {len(human_metadata)}, VLMs: {len(vlm_metadata)})")

    return combined_embeddings, combined_metadata

def apply_pca(embeddings, variance_ratio=0.95):
    """Apply PCA to reduce dimensionality of embeddings while retaining specified variance."""
    pca = PCA(n_components=variance_ratio)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"PCA applied: Reduced dimensions to {reduced_embeddings.shape[1]} components.")
    return reduced_embeddings


def compute_pairwise_scores_with_cache(embeddings, metadata, cache_path, max_workers=4):
    """Compute pairwise cosine similarity scores with incremental caching.
    
    If a cache file exists, loads it and only computes scores for agent
    pairs that are missing (e.g., because a new agent was added).
    """
    existing_df = None
    if os.path.exists(cache_path):
        print(f"\nLoading cached pairwise scores from {cache_path}...")
        existing_df = pd.read_csv(cache_path)
        
        # Check if there are new agents
        current_agents = set(metadata["AGENT"].unique())
        cached_agents = set(existing_df["AGENT_I"].unique()) | set(existing_df["AGENT_J"].unique())
        new_agents = current_agents - cached_agents
        
        if not new_agents:
            print(f"  All {len(current_agents)} agents accounted for — using cache")
            return existing_df
        
        print(f"  \u26a1 Detected {len(new_agents)} new agent(s): {sorted(new_agents)}")
        print(f"  Computing only pairs involving new agents...")
    else:
        print(f"\nNo cache found at {cache_path} — computing all pairwise scores...")
        new_agents = set(metadata["AGENT"].unique())
    
    # If no cache or new agents detected, compute missing pairs
    # For simplicity with the embedding-based approach, compute all if no cache
    print(f"\nComputing pairwise scores for new agents only...")
    total_agent_pairs = 0
    # Only compute pairs involving at least one new agent
    agents = utils.get_agent_groups()["all"]
    to_compute = []
    #drop duplicates for video agent and question num
    for (video, question_num), group in tqdm(
        metadata.groupby(["VIDEO", "QUESTION_NUM"]),
        total=metadata.groupby(["VIDEO", "QUESTION_NUM"]).ngroups
    ):
        available_agents = group["AGENT"].unique()
        for agent_i, agent_j in combinations_with_replacement(agents, 2):
            if agent_i not in available_agents or agent_j not in available_agents:
                continue
            total_agent_pairs += 1
            to_compute.append({
                "VIDEO": video,
                "QUESTION_NUM": question_num,
                "AGENT_I": agent_i,
                "AGENT_J": agent_j,
                "BLOCK": group["BLOCK"].iloc[0],
            })

    #convert to dataframe for easier processing
    to_compute_df = pd.DataFrame(to_compute)
    new_scores = []
    for _, row in tqdm(to_compute_df.iterrows(), total=len(to_compute_df), desc="Computing new pairwise scores"):
        emb_i = embeddings[(row["AGENT_I"], row["VIDEO"], row["QUESTION_NUM"],1)]
        emb_j = embeddings[(row["AGENT_J"], row["VIDEO"], row["QUESTION_NUM"],1)]
        
        #better compute it manually
            #normalize the embeddings and compute the cosine similarity
        emb_i = emb_i / np.linalg.norm(emb_i)
        emb_j = emb_j / np.linalg.norm(emb_j)
        cosine_score = np.dot(emb_i, emb_j)
        
        new_scores.append({
            "VIDEO": row["VIDEO"],
            "QUESTION_NUM": row["QUESTION_NUM"],
            "AGENT_I": row["AGENT_I"],
            "AGENT_J": row["AGENT_J"],
            "BLOCK": row["BLOCK"],
            "SCORE": cosine_score
        })
        #also add the reverse pair for symmetry
        new_scores.append({
            "VIDEO": row["VIDEO"],
            "QUESTION_NUM": row["QUESTION_NUM"],
            "AGENT_I": row["AGENT_J"],
            "AGENT_J": row["AGENT_I"],
            "BLOCK": row["BLOCK"],
            "SCORE": cosine_score
        })
        
    #nopw add the score with same using agents
    for agent in agents:
        for video, question_num in metadata.groupby(["VIDEO", "QUESTION_NUM"]).groups.keys():
            if (agent, video, question_num) not in embeddings:
                continue
            new_scores.append({
                "VIDEO": video,
                "QUESTION_NUM": question_num,
                "AGENT_I": agent,
                "AGENT_J": agent,
                "BLOCK": metadata[(metadata["VIDEO"] == video) & (metadata["QUESTION_NUM"] == question_num)]["BLOCK"].iloc[0],
                "SCORE": 1.0
            })
            

    new_df = pd.DataFrame(new_scores)
    pairwise_scores = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"  Added {len(new_scores)} new pair scores")
    
    # Save to cache
    print(f"\nSaving pairwise scores to cache at {cache_path}...")
    pairwise_scores.to_csv(cache_path, index=False)
    
    return pairwise_scores

def main():
    """Main execution: check cache, process embeddings, compute similarity, and generate heatmaps."""
    print("\n=== Cosine Similarity Heatmap Analysis ===")

    # Parse arguments for VLM processing mode
    parser = argparse.ArgumentParser(description="Heatmap analysis for cosine similarity of embeddings")
    parser.add_argument("--vlm-mode", choices=["mean", "first"], default="first",
                        help="How to process VLM embeddings: 'mean' (average) or 'first'")
    parser.add_argument("--apply-pca", action="store_true",
                        help="Apply PCA for dimensionality reduction with 95%% explained variance.")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Number of workers for parallel pairwise score computation (default: 10)")
    args = parser.parse_args()

    print(f"\nProcessing VLM embeddings using mode: {args.vlm_mode}\n")

    # Check for embedding cache (keyed or legacy)
    check_embedding_cache(None)

    # Load embeddings via the keyed cache (aligns to current CSV)
    
    metadata = utils.load_answers()
    keyed_path = config.EMBEDDING_CACHE["keyed"]
    legacy_path = config.EMBEDDING_CACHE["legacy_npy"]

    if os.path.exists(keyed_path):
        import pickle
        from src.old.embed_analysis import _make_embed_key
        with open(keyed_path, "rb") as f:
            cache = pickle.load(f)
        keys = [_make_embed_key(row) for _, row in metadata.iterrows()]
        missing = [k for k in keys if k not in cache]
        if missing:
            print(f"  ⚠ {len(missing)} rows missing from keyed cache — run embed_analysis.py first")
            sys.exit(1)
        embeddings = np.vstack([cache[k] for k in keys])
        dict_embeddings = {k: cache[k] for k in keys}  # For easy lookup by (VIDEO, QUESTION_NUM, AGENT)
        print(f"✓ Loaded embeddings from keyed cache: {keyed_path} with {len(cache)} entries")
        #print(keys)
    else:
        embeddings = np.load(legacy_path)

    # Ensure 'BLOCK' column exists in metadata
    if "BLOCK" not in metadata.columns:
        metadata = utils.compute_blocks(metadata)

    # Process embeddings based on VLM mode
    embeddings, metadata = process_vlm_embeddings(embeddings, metadata, mode=args.vlm_mode)



    # Apply PCA if specified
    if args.apply_pca:
        print("\nApplying PCA for dimensionality reduction...")
        embeddings = apply_pca(embeddings)

    # Compute pairwise cosine similarity scores with caching
    pairwise_cache_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "pairwise_scores_cache.csv")
    pairwise_df = compute_pairwise_scores_with_cache(dict_embeddings, metadata, pairwise_cache_path, max_workers=args.max_workers)

    # Generate similarity heatmaps
    generate_score_similarity_heatmaps(pairwise_df)

    print("\n✓ Cosine similarity and agreement heatmap analysis complete!\n")

if __name__ == "__main__":
    main()