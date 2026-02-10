"""Embedding analysis with UMAP and PCA dimensionality reduction."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from umap import UMAP
from src import config, utils


EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBED_MODEL_SMALL = "all-MiniLM-L6-v2"  # For testing or smaller datasets
BATCH_SIZE = 64  # Adjust based on your GPU/CPU capabilities

def generate_embeddings(df: pd.DataFrame, use_cache: bool = False) -> tuple:
    """
    Generate sentence embeddings for all answers.
    
    Args:
        use_cache: If True, try to load cached embeddings or save after generation
                   WARNING: Generation is expensive! Use cache carefully.
    """
    print("\n=== Generating Embeddings ===")
    
    cache_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "embeddings_cache.npy")
    
    # Try to load from cache if enabled
    if use_cache and os.path.exists(cache_path):
        print(f"⚠ Loading cached embeddings from: {os.path.basename(cache_path)}")
        embeddings = np.load(cache_path)
        print(f"✓ Loaded cached embeddings of shape {embeddings.shape}")
        return embeddings, df
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer(
        EMBED_MODEL,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": "bfloat16"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    print("✓ Model loaded")
    
    # Get answers as list
    answers = df["ANSWER"].astype(str).tolist()
    
    # Generate embeddings (EXPENSIVE!)
    print(f"⚠ Encoding {len(answers)} answers... (this may take time)")
    embeddings = model.encode(answers, show_progress_bar=True, batch_size=BATCH_SIZE)
    print(f"✓ Generated embeddings of shape {embeddings.shape}")
    
    # Save to cache if enabled
    if use_cache:
        utils.ensure_output_dir(config.OUTPUT_EMBEDDINGS_DIR)
        np.save(cache_path, embeddings)
        print(f"✓ Cached embeddings to: {os.path.basename(cache_path)}")
    
    return embeddings, df


def reduce_and_plot_umap(embeddings, df: pd.DataFrame) -> None:
    """Reduce embeddings with UMAP and create scatter plot."""
    print("\n=== UMAP Reduction ===")
    
    # Apply UMAP
    print("Applying UMAP reduction (2D)...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)
    print("✓ UMAP reduction complete")
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by agent type
    for agent in df["AGENT"].unique():
        mask = df["AGENT"] == agent
        color = utils.get_agent_color(agent)
        
        label = agent
        if agent in config.LIMA_AGENTS:
            label = "Lima Humans" if agent == config.LIMA_AGENTS[0] else None
        elif agent in config.NYC_AGENTS:
            label = "NYC Humans" if agent == config.NYC_AGENTS[0] else None
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_title("UMAP Projection of Answer Embeddings", fontsize=16, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "umap_embeddings.png")
    utils.ensure_output_dir(config.OUTPUT_EMBEDDINGS_DIR)
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")


def reduce_and_plot_pca(embeddings, df: pd.DataFrame) -> None:
    """Reduce embeddings with PCA and create 2D scatter plot."""
    print("\n=== PCA Reduction ===")
    
    # Apply PCA (2D)
    print("Applying PCA reduction (2D)...")
    pca_2d = PCA(n_components=2, random_state=42)
    embeddings_2d = pca_2d.fit_transform(embeddings)
    variance_2d = pca_2d.explained_variance_ratio_
    print(f"✓ PCA 2D complete (explained variance: {variance_2d.sum():.2%})")
    
    # Plot 2D
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for agent in df["AGENT"].unique():
        mask = df["AGENT"] == agent
        color = utils.get_agent_color(agent)
        
        label = agent
        if agent in config.LIMA_AGENTS:
            label = "Lima Humans" if agent == config.LIMA_AGENTS[0] else None
        elif agent in config.NYC_AGENTS:
            label = "NYC Humans" if agent == config.NYC_AGENTS[0] else None
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color,
            label=label,
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_title("PCA Projection of Answer Embeddings (2D)", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"PC1 ({variance_2d[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({variance_2d[1]:.1%})", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "pca_embeddings_2d.png")
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")


def main(use_cache: bool = True):
    """
    Main execution: generate embeddings and visualizations.
    
    Args:
        use_cache: If True, cache embeddings to disk (saves time on reruns)
                   Default False - always regenerate (safer but slower)
    """
    print("=" * 60)
    print("EMBEDDING ANALYSIS (UMAP & PCA)")
    print("=" * 60)
    
    if use_cache:
        print("\n⚠ Cache mode ENABLED - will save/load embeddings")
    else:
        print("\n⚠ Cache mode DISABLED - will regenerate embeddings (expensive!)")
        print("   Tip: Use use_cache=True to enable caching")
    
    # Load answers
    df = utils.load_answers()
    
    # Generate embeddings
    embeddings, df = generate_embeddings(df, use_cache=use_cache)
    
    # UMAP visualization
    reduce_and_plot_umap(embeddings, df)
    
    # PCA visualizations
    reduce_and_plot_pca(embeddings, df)
    
    print("\n" + "=" * 60)
    print("✓ Embedding analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


