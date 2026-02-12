"""Heatmap analysis for cosine similarity of embeddings."""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.heatmap_common import plot_similarity_heatmap, create_agreement_matrix, plot_agreement_heatmap
from src import config, utils
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA

def check_embedding_cache(cache_path):
    """Check if the embedding cache exists (keyed or legacy)."""
    keyed_path = config.EMBEDDING_CACHE["keyed"]
    legacy_path = config.EMBEDDING_CACHE["legacy_npy"]
    if not os.path.exists(keyed_path) and not os.path.exists(legacy_path):
        print("\n⚠ Embedding cache not found! Please run `embed_analysis.py` to generate embeddings first.\n")
        sys.exit(1)

def compute_cosine_similarity(embeddings):
    """Compute pairwise cosine similarity from embeddings."""
    # Cosine similarity is 1 - cosine distance
    similarity_matrix = 1 - cdist(embeddings, embeddings, metric="cosine")
    return pd.DataFrame(similarity_matrix)

def generate_cosine_similarity_heatmaps(embeddings, metadata):
    """Generate heatmaps for cosine similarity by block."""
    output_dir = config.OUTPUT_EMBEDDINGS_DIR
    utils.ensure_output_dir(output_dir)

    for block in sorted(metadata["BLOCK"].unique()):
        block_mask = metadata["BLOCK"] == block
        block_embeddings = embeddings[block_mask]
        block_metadata = metadata[block_mask]

        # Compute cosine similarity
        similarity_matrix = compute_cosine_similarity(block_embeddings)
        similarity_matrix.index = block_metadata["AGENT"].values
        similarity_matrix.columns = block_metadata["AGENT"].values

        # Plot similarity heatmap
        similarity_path = os.path.join(output_dir, f"cosine_similarity_block{block}.png")
        plot_similarity_heatmap(
            similarity_matrix,
            title=f"Cosine Similarity - Block {block}",
            output_path=similarity_path,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0
        )

def generate_cosine_agreement_heatmaps(pairwise_df: pd.DataFrame, metadata: pd.DataFrame):
    """Generate agreement heatmaps for cosine similarity by block."""
    output_dir = config.OUTPUT_EMBEDDINGS_DIR
    utils.ensure_output_dir(output_dir)

    for block in sorted(metadata["BLOCK"].unique()):
        # Compute agreement matrix
        agreement_matrix = create_agreement_matrix(
            pairwise_df,
            block,
            score_column="COSINE_SCORE",
            threshold=0.5  # Example threshold for agreement
        )

        # Plot agreement heatmap
        agreement_path = os.path.join(output_dir, f"cosine_agreement_block{block}.png")
        plot_agreement_heatmap(
            agreement_matrix,
            title=f"Cosine Agreement - Block {block}",
            output_path=agreement_path,
            cmap="RdYlGn",
            use_group_colors=True
        )

def process_vlm_embeddings(embeddings: np.ndarray, metadata: pd.DataFrame, mode: str = "mean") -> tuple:
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

    return combined_embeddings, combined_metadata

def apply_pca(embeddings, variance_ratio=0.95):
    """Apply PCA to reduce dimensionality of embeddings while retaining specified variance."""
    pca = PCA(n_components=variance_ratio)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"PCA applied: Reduced dimensions to {reduced_embeddings.shape[1]} components.")
    return reduced_embeddings

def compute_score(i, row_i, metadata, embeddings):
    """Compute cosine similarity scores for a given row."""
    scores = []
    for j, row_j in metadata.iterrows():
        if i >= j:
            continue
        score = 1 - cdist([embeddings[i]], [embeddings[j]], metric="cosine")[0][0]
        scores.append({
            "BLOCK": row_i["BLOCK"],
            "AGENT_I": row_i["AGENT"],
            "AGENT_J": row_j["AGENT"],
            "COSINE_SCORE": score
        })
    return scores

def compute_pairwise_scores(embeddings, metadata):
    """Compute pairwise cosine similarity scores using multithreading with tqdm progress."""
    pairwise_scores = []

    print("\nComputing pairwise cosine similarity scores with multithreading...")
    import concurrent
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        futures = {executor.submit(compute_score, i, row_i, metadata, embeddings): i for i, row_i in metadata.iterrows()}
        with tqdm(total=len(futures), desc="Rows processed") as pbar:
            for future in as_completed(futures):
                pairwise_scores.extend(future.result())
                pbar.update(1)

    return pd.DataFrame(pairwise_scores)

def compute_pairwise_scores_with_cache(embeddings, metadata, cache_path):
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
    
    # If no cache or new agents detected, compute missing pairs
    # For simplicity with the embedding-based approach, compute all if no cache
    if existing_df is None:
        pairwise_scores = compute_pairwise_scores(embeddings, metadata)
    else:
        # Only compute pairs involving at least one new agent
        new_scores = []
        for i, row_i in metadata.iterrows():
            for j, row_j in metadata.iterrows():
                if i >= j:
                    continue
                # Skip if both agents are already cached
                if row_i["AGENT"] not in new_agents and row_j["AGENT"] not in new_agents:
                    continue
                score = 1 - cdist([embeddings[i]], [embeddings[j]], metric="cosine")[0][0]
                new_scores.append({
                    "BLOCK": row_i["BLOCK"],
                    "AGENT_I": row_i["AGENT"],
                    "AGENT_J": row_j["AGENT"],
                    "COSINE_SCORE": score
                })
        
        if new_scores:
            new_df = pd.DataFrame(new_scores)
            pairwise_scores = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"  Added {len(new_scores)} new pair scores")
        else:
            pairwise_scores = existing_df
    
    # Save to cache
    print(f"\nSaving pairwise scores to cache at {cache_path}...")
    pairwise_scores.to_csv(cache_path, index=False)
    
    return pairwise_scores

def main():
    """Main execution: check cache, process embeddings, compute similarity, and generate heatmaps."""
    print("\n=== Cosine Similarity Heatmap Analysis ===")

    # Check for embedding cache (keyed or legacy)
    check_embedding_cache(None)

    # Load embeddings via the keyed cache (aligns to current CSV)
    metadata = utils.load_answers()

    # Try keyed cache first, fall back to legacy .npy
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
    else:
        embeddings = np.load(legacy_path)

    # Ensure 'BLOCK' column exists in metadata
    if "BLOCK" not in metadata.columns:
        metadata = utils.compute_blocks(metadata)

    # Parse arguments for VLM processing mode
    parser = argparse.ArgumentParser(description="Heatmap analysis for cosine similarity of embeddings")
    parser.add_argument("--vlm-mode", choices=["mean", "first"], default="mean",
                        help="How to process VLM embeddings: 'mean' (average) or 'first' (use first response)")
    parser.add_argument("--apply-pca", action="store_true",
                        help="Apply PCA for dimensionality reduction with 95%% explained variance.")
    args = parser.parse_args()

    print(f"\nProcessing VLM embeddings using mode: {args.vlm_mode}\n")

    # Process embeddings based on VLM mode
    embeddings, metadata = process_vlm_embeddings(embeddings, metadata, mode=args.vlm_mode)

    # Apply PCA if specified
    if args.apply_pca:
        print("\nApplying PCA for dimensionality reduction...")
        embeddings = apply_pca(embeddings)

    # Compute pairwise cosine similarity scores with caching
    pairwise_cache_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "pairwise_scores_cache.csv")
    pairwise_df = compute_pairwise_scores_with_cache(embeddings, metadata, pairwise_cache_path)

    # Generate similarity heatmaps
    generate_cosine_similarity_heatmaps(embeddings, metadata)

    # Generate agreement heatmaps
    generate_cosine_agreement_heatmaps(pairwise_df, metadata)

    print("\n✓ Cosine similarity and agreement heatmap analysis complete!\n")

if __name__ == "__main__":
    main()