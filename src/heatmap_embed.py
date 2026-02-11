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
    """Check if the embedding cache exists."""
    if not os.path.exists(cache_path):
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
    """Compute pairwise cosine similarity scores with caching."""
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"\nLoading cached pairwise scores from {cache_path}...")
        return pd.read_csv(cache_path)

    # Compute pairwise scores
    pairwise_scores = compute_pairwise_scores(embeddings, metadata)

    # Save to cache
    print(f"\nSaving pairwise scores to cache at {cache_path}...")
    pairwise_scores.to_csv(cache_path, index=False)

    return pairwise_scores

def main():
    """Main execution: check cache, process embeddings, compute similarity, and generate heatmaps."""
    print("\n=== Cosine Similarity Heatmap Analysis ===")

    # Check for embedding cache
    cache_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "embeddings_cache.npy")
    check_embedding_cache(cache_path)

    # Load embeddings and metadata
    embeddings = np.load(cache_path)
    metadata = utils.load_answers()

    # Ensure 'BLOCK' column exists in metadata
    if "BLOCK" not in metadata.columns:
        metadata = utils.compute_blocks(metadata)

    # Parse arguments for VLM processing mode
    parser = argparse.ArgumentParser(description="Heatmap analysis for cosine similarity of embeddings")
    parser.add_argument("--vlm-mode", choices=["mean", "first"], default="mean",
                        help="How to process VLM embeddings: 'mean' (average) or 'first' (use first response)")
    parser.add_argument("--apply-pca", action="store_true",
                        help="Apply PCA for dimensionality reduction with 95% explained variance.")
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