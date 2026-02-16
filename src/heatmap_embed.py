"""Heatmap analysis for cosine similarity of embeddings."""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.heatmap_common import plot_similarity_heatmap, create_agreement_matrix
from src import config, utils
import argparse
from itertools import combinations
from tqdm import tqdm
from sklearn.decomposition import PCA

def check_embedding_cache(cache_path):
    """Check if the embedding cache exists (keyed or legacy)."""
    keyed_path = config.EMBEDDING_CACHE["keyed"]
    legacy_path = config.EMBEDDING_CACHE["legacy_npy"]
    if not os.path.exists(keyed_path) and not os.path.exists(legacy_path):
        print("\n⚠ Embedding cache not found! Please run `embed_analysis.py` to generate embeddings first.\n")
        sys.exit(1)

def generate_score_similarity_heatmaps(metadata):
    """Generate heatmaps for cosine similarity by block."""
    output_dir = config.OUTPUT_EMBEDDINGS_DIR
    utils.ensure_output_dir(output_dir)
    region_labels = {"both": "All", "lima": "Lima", "nyc": "NYC"}

    # Generate color map
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cmap", ["#ffffff", "#963fc9"])

    metadata = metadata.copy()
    if "VIDEO_REGION" not in metadata.columns:
        metadata["VIDEO_REGION"] = metadata["VIDEO"].apply(utils.infer_video_region)

    for block in sorted(metadata["BLOCK"].unique()):
        for region in ["both", "lima", "nyc"]:
            block_mask = metadata["BLOCK"] == block
            if region != "both":
                block_mask = block_mask & (metadata["VIDEO_REGION"] == region)

            block_metadata = metadata[block_mask]
            if len(block_metadata) < 1:
                print(f"⚠ No embedding rows for block={block}, region={region}")
                continue
            
            df_avg = (
                block_metadata
                .groupby(["AGENT_I", "AGENT_J"], as_index=False)
                .SCORE.mean()
            )
            
            matrix = df_avg.pivot(
                index="AGENT_I",
                columns="AGENT_J",
                values="SCORE"
            )
            #now pivot? or just compute pairwise cosine similarity for this block and region?
            # Plot similarity heatmap
            if region == "both":
                similarity_path = os.path.join(output_dir, f"cosine_score_block{block}.png")
            else:
                similarity_path = os.path.join(output_dir, f"cosine_score_block{block}_{region}.png")

            region_title = "" if region == "both" else f" ({region_labels[region]})"
            plot_similarity_heatmap(
                matrix,
                title=f"Cosine Score Similarity - Block {block}{region_title}",
                color_label="Cosine Similarity",
                output_path=similarity_path,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0
            )
            
            # Agreement heatmap
            agreement_matrix = create_agreement_matrix(
                block_metadata,
                block,
                score_column="SCORE",
                threshold=0.5,
                video_region=region,
            )

            if region == "both":
                agreement_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, f"cosine_agreement_block{block}.png")
            else:
                agreement_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, f"cosine_agreement_block{block}_{region}.png")

            plot_similarity_heatmap(
                agreement_matrix,
                title=f"Cosine Score Agreement - Block {block}{region_title}",
                color_label="Agreement (%)",
                output_path=agreement_path,
                cmap=cmap,
                vmin=0,
                vmax=100,
                fmt="%d",
                use_group_colors=True
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
        for agent_i, agent_j in combinations(agents, 2):
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
    #now compute cosine similarity for these pairs using the index to get the embeddings
    new_scores = []
    for _, row in tqdm(to_compute_df.iterrows(), total=len(to_compute_df), desc="Computing new pairwise scores"):
        emb_i = embeddings[(row["AGENT_I"], row["VIDEO"], row["QUESTION_NUM"] )]
        emb_j = embeddings[(row["AGENT_J"], row["VIDEO"], row["QUESTION_NUM"] )]
        cosine_score = 1 - cdist([emb_i], [emb_j], metric="cosine")[0][0]
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
        from src.embed_analysis import _make_embed_key
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