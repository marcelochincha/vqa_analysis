"""Embedding analysis with UMAP and PCA dimensionality reduction."""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for plotting
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


def process_vlm_embeddings(embeddings: np.ndarray, df: pd.DataFrame, mode: str = "mean") -> tuple:
    """
    Process VLM embeddings: either average 20 repetitions or keep only first.
    
    Args:
        embeddings: All embeddings (N x dim)
        df: DataFrame with all rows
        mode: "mean" to average VLM repetitions, "first" to keep only first
    
    Returns:
        Processed embeddings and dataframe
    """
    print(f"\n=== Processing VLMs (mode={mode}) ===")
    
    if mode not in ["mean", "first"]:
        raise ValueError(f"Invalid mode: {mode}. Use 'mean' or 'first'")
    
    # Add block info
    df = utils.compute_blocks(df)
    
    # Add video sector (Lima: <=100, NYC: >100)
    df['VIDEO_NUM'] = df['VIDEO'].str.extract(r'_(\d+)$')[0].astype(int)
    df['VIDEO_SECTOR'] = df['VIDEO_NUM'].apply(lambda x: 'Lima' if x <= 100 else 'NYC')
    
    # Separate humans and VLMs
    human_mask = df["AGENT"].isin(config.HUMAN_AGENTS)
    vlm_mask = df["AGENT"].isin(config.VLM_AGENTS)
    
    # Keep all humans (they only have 1 answer per question)
    df_humans = df[human_mask].copy()
    embeddings_humans = embeddings[human_mask]
    
    print(f"✓ Humans: {len(df_humans)} rows")
    
    # Process VLMs
    df_vlms = df[vlm_mask].copy()
    embeddings_vlms = embeddings[vlm_mask]
    
    if mode == "first":
        # Keep only first repetition for each (VIDEO, QUESTION_NUM, AGENT)
        df_vlms_proc = df_vlms.drop_duplicates(subset=["VIDEO", "QUESTION_NUM", "AGENT"], keep="first")
        first_indices = df_vlms_proc.index
        # Get corresponding embeddings
        vlm_indices_in_full = np.where(vlm_mask)[0]
        first_positions = [np.where(df_vlms.index == idx)[0][0] for idx in first_indices]
        embeddings_vlms_proc = embeddings_vlms[first_positions]
        print(f"✓ VLMs (first only): {len(df_vlms_proc)} rows")
    else:  # mean
        # Group by (VIDEO, QUESTION_NUM, AGENT) and average embeddings
        grouped = df_vlms.groupby(["VIDEO", "QUESTION_NUM", "AGENT"])
        
        vlm_data = []
        vlm_embeds = []
        
        for (video, qnum, agent), group in grouped:
            # Get indices of this group in the VLM subset
            group_positions = [np.where(df_vlms.index == idx)[0][0] for idx in group.index]
            # Average embeddings
            mean_embed = embeddings_vlms[group_positions].mean(axis=0)
            vlm_embeds.append(mean_embed)
            
            # Keep first row's metadata
            first_row = group.iloc[0].to_dict()
            first_row["ANSWER"] = f"[MEAN of {len(group)} reps]"
            vlm_data.append(first_row)
        
        df_vlms_proc = pd.DataFrame(vlm_data)
        embeddings_vlms_proc = np.array(vlm_embeds)
        print(f"✓ VLMs (mean): {len(df_vlms_proc)} rows")
    
    # Combine humans + processed VLMs
    df_final = pd.concat([df_humans, df_vlms_proc], ignore_index=True)
    embeddings_final = np.vstack([embeddings_humans, embeddings_vlms_proc])
    
    print(f"✓ Total processed: {len(df_final)} rows")
    
    return embeddings_final, df_final


def get_agent_marker(agent: str) -> str:
    """Get marker style for agent type."""
    if agent in config.VLM_AGENTS:
        return "o"  # Circle for VLMs
    elif agent in config.LIMA_AGENTS:
        return "s"  # Square for Lima
    elif agent in config.NYC_AGENTS:
        return "^"  # Triangle for NYC
    return "x"


def reduce_and_plot_umap_by_block(embeddings, df: pd.DataFrame) -> None:
    """Create UMAP plots separated by block and video sector."""
    print("\n=== UMAP by Block and Sector ===")
    
    for block in sorted(df["BLOCK"].unique()):
        # Fit UMAP once per block with ALL videos (Lima + NYC)
        block_mask = df["BLOCK"] == block
        embeddings_block = embeddings[block_mask]
        df_block = df[block_mask]
        
        print(f"Block {block}: Fitting UMAP with {len(df_block)} total samples")
        
        # Apply UMAP to entire block
        n_neighbors = min(15, len(df_block)-1)
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        embeddings_2d_block = reducer.fit_transform(embeddings_block)
        
        # Now plot each sector separately in the same embedding space
        for sector in ['Lima', 'NYC']:
            sector_mask = df_block["VIDEO_SECTOR"] == sector
            embeddings_2d = embeddings_2d_block[sector_mask]
            df_subset = df_block[sector_mask]
            
            print(f"  {sector}: {len(df_subset)} samples")
            
            if len(df_subset) < 1:
                print(f"    Skipping (no data)")
                continue
            
            # Plot with individual agent colors and markers
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Collect handles for grouped legend
            handles_vlm = []
            handles_lima = []
            handles_nyc = []
            
            # Plot VLMs (orange tones)
            for agent in config.VLM_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        c=config.AGENT_COLORS_MAP[agent],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_vlm.append(scatter)
            
            # Plot Lima humans (light blue tones)
            for agent in config.LIMA_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        c=config.AGENT_COLORS_MAP[agent],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_lima.append(scatter)
            
            # Plot NYC humans (dark blue tones)
            for agent in config.NYC_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        c=config.AGENT_COLORS_MAP[agent],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_nyc.append(scatter)
            
            # Create grouped legend: VLMs | GRUPO LIMA | GRUPO NYC
            all_handles = []
            all_labels = []
            
            if handles_vlm:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ VLMs ━━━',
                                             markerfacecolor='orange', markersize=0, linestyle='None'))
                all_labels.append('━━━ VLMs ━━━')
                for h in handles_vlm:
                    all_handles.append(h)
                    all_labels.append(h.get_label())
            
            if handles_lima:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ GRUPO LIMA ━━━',
                                             markerfacecolor='blue', markersize=0, linestyle='None'))
                all_labels.append('━━━ GRUPO LIMA ━━━')
                for h in handles_lima:
                    all_handles.append(h)
                    all_labels.append(h.get_label())
            
            if handles_nyc:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ GRUPO NYC ━━━',
                                             markerfacecolor='darkblue', markersize=0, linestyle='None'))
                all_labels.append('━━━ GRUPO NYC ━━━')
                for h in handles_nyc:
                    all_handles.append(h)
                    all_labels.append(h.get_label())
            
            ax.set_title(f"UMAP - Block {block} - Videos {sector} (Q{(block-1)*5+1}-Q{block*5})", 
                        fontsize=14, fontweight="bold")
            ax.set_xlabel("UMAP Dimension 1", fontsize=11)
            ax.set_ylabel("UMAP Dimension 2", fontsize=11)
            ax.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(1, 0.5), 
                     frameon=True, fontsize=8, ncol=1)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            sector_label = sector.lower()
            output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, 
                                      f"umap_block{block}_{sector_label}.png")
            plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
            plt.close()
            
            print(f"  Saved: {os.path.basename(output_path)}")


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


def reduce_and_plot_pca_by_block(embeddings, df: pd.DataFrame) -> None:
    """Create PCA plots separated by block and video sector."""
    print("\n=== PCA by Block and Sector ===")
    
    for block in sorted(df["BLOCK"].unique()):
        # Fit PCA once per block with ALL videos (Lima + NYC)
        block_mask = df["BLOCK"] == block
        embeddings_block = embeddings[block_mask]
        df_block = df[block_mask]
        
        print(f"Block {block}: Fitting PCA with {len(df_block)} total samples")
        
        # Apply PCA to entire block
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d_block = pca.fit_transform(embeddings_block)
        variance = pca.explained_variance_ratio_
        
        # Now plot each sector separately in the same embedding space
        for sector in ['Lima', 'NYC']:
            sector_mask = df_block["VIDEO_SECTOR"] == sector
            embeddings_2d = embeddings_2d_block[sector_mask]
            df_subset = df_block[sector_mask]
            
            print(f"  {sector}: {len(df_subset)} samples")
            
            if len(df_subset) < 1:
                print(f"    Skipping (no data)")
                continue
            
            # Plot with individual agent colors and markers
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Collect handles for grouped legend
            handles_vlm = []
            handles_lima = []
            handles_nyc = []
            
            # Plot VLMs (orange tones)
            for agent in config.VLM_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        c=config.AGENT_COLORS_MAP[agent],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_vlm.append(scatter)
            
            # Plot Lima humans (light blue tones)
            for agent in config.LIMA_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        c=config.AGENT_COLORS_MAP[agent],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_lima.append(scatter)
            
            # Plot NYC humans (dark blue tones)
            for agent in config.NYC_AGENTS:
                if agent in df_subset["AGENT"].values:
                    agent_mask = df_subset["AGENT"] == agent
                    mask_array = agent_mask.values
                    scatter = ax.scatter(
                        embeddings_2d[mask_array, 0],
                        embeddings_2d[mask_array, 1],
                        c=config.AGENT_COLORS_MAP[agent],
                        marker=config.AGENT_MARKERS_MAP[agent],
                        label=agent,
                        alpha=0.7,
                        s=60,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    handles_nyc.append(scatter)
            
            # Create grouped legend: VLMs | GRUPO LIMA | GRUPO NYC
            all_handles = []
            all_labels = []
            
            if handles_vlm:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ VLMs ━━━',
                                             markerfacecolor='orange', markersize=0, linestyle='None'))
                all_labels.append('━━━ VLMs ━━━')
                for h in handles_vlm:
                    all_handles.append(h)
                    all_labels.append(h.get_label())
            
            if handles_lima:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ GRUPO LIMA ━━━',
                                             markerfacecolor='blue', markersize=0, linestyle='None'))
                all_labels.append('━━━ GRUPO LIMA ━━━')
                for h in handles_lima:
                    all_handles.append(h)
                    all_labels.append(h.get_label())
            
            if handles_nyc:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ GRUPO NYC ━━━',
                                             markerfacecolor='darkblue', markersize=0, linestyle='None'))
                all_labels.append('━━━ GRUPO NYC ━━━')
                for h in handles_nyc:
                    all_handles.append(h)
                    all_labels.append(h.get_label())
            
            ax.set_title(f"PCA - Block {block} - Videos {sector} (Q{(block-1)*5+1}-Q{block*5})", 
                        fontsize=14, fontweight="bold")
            ax.set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=11)
            ax.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=11)
            ax.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(1, 0.5), 
                     frameon=True, fontsize=8, ncol=1)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            sector_label = sector.lower()
            output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, 
                                      f"pca_block{block}_{sector_label}.png")
            plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
            plt.close()
            
            print(f"  Saved: {os.path.basename(output_path)}")


def main(use_cache: bool = True, vlm_mode: str = "mean"):
    """
    Main execution: generate embeddings and visualizations.
    
    Args:
        use_cache: If True, cache embeddings to disk (saves time on reruns)
        vlm_mode: How to process VLMs - "mean" (average 20 reps) or "first" (only 1st)
    """
    print("=" * 60)
    print("EMBEDDING ANALYSIS (UMAP & PCA BY BLOCKS)")
    print("=" * 60)
    
    if use_cache:
        print("\n⚠ Cache mode ENABLED - will save/load embeddings")
    else:
        print("\n⚠ Cache mode DISABLED - will regenerate embeddings (expensive!)")
    
    print(f"⚠ VLM mode: {vlm_mode.upper()} (use --mean-vlm or --first-vlm to change)\n")
    
    # Load answers
    df = utils.load_answers()
    
    # Generate embeddings for ALL data (VLMs 20 reps + humans)
    embeddings_all, df_all = generate_embeddings(df, use_cache=use_cache)
    
    # Process VLMs (mean or first) for plotting
    embeddings, df = process_vlm_embeddings(embeddings_all, df_all, mode=vlm_mode)
    
    # Generate plots by block
    reduce_and_plot_umap_by_block(embeddings, df)
    reduce_and_plot_pca_by_block(embeddings, df)
    
    print("\n" + "=" * 60)
    print("✓ Embedding analysis complete!")
    print(f"✓ Generated 16 plots (4 blocks x 2 sectors x 2 methods) in: {config.OUTPUT_EMBEDDINGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding analysis with UMAP and PCA")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache (regenerate)")
    parser.add_argument("--mean-vlm", action="store_true", help="Average VLM 20 repetitions (default)")
    parser.add_argument("--first-vlm", action="store_true", help="Use only first VLM repetition")
    
    args = parser.parse_args()
    
    # Determine VLM mode
    if args.first_vlm:
        vlm_mode = "first"
    else:
        vlm_mode = "mean"  # Default
    
    main(use_cache=not args.no_cache, vlm_mode=vlm_mode)


