"""Embedding analysis with UMAP and PCA dimensionality reduction."""
import os
import argparse
import pickle
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


# ============================================================================
# Keyed embedding cache (incremental)
# ============================================================================

def _make_embed_key(row) -> tuple:
    """Create a unique cache key for an answer row."""
    return (str(row["AGENT"]), str(row["VIDEO"]), int(row["QUESTION_NUM"]), int(row["REPETITION"]))


def load_embedding_cache(cache_path: str) -> dict:
    """Load keyed embedding cache: dict[(agent,video,qnum)] -> np.ndarray."""
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"✓ Loaded keyed embedding cache: {len(cache):,} entries")
    return cache


def save_embedding_cache(cache: dict, cache_path: str) -> None:
    """Save keyed embedding cache to disk."""
    utils.ensure_output_dir(os.path.dirname(cache_path))
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved keyed embedding cache: {len(cache):,} entries")


def migrate_npy_to_keyed_cache(npy_path: str, keyed_path: str, df: pd.DataFrame) -> dict:
    """Migrate legacy monolithic .npy cache to keyed dict cache.
    
    The .npy stores embeddings aligned by row order of the CSV used to
    create it.  We pair each row of ``df`` with the corresponding
    embedding vector and build the keyed dict.
    """
    print("\n--- Migrating legacy .npy embedding cache to keyed format ---")
    embeddings = np.load(npy_path)
    
    if len(embeddings) != len(df):
        print(f"  ⚠ Shape mismatch: .npy has {len(embeddings)} rows, CSV has {len(df)} rows")
        print("    Migration will use min(npy_rows, csv_rows)")
    
    n = min(len(embeddings), len(df))
    cache = {}
    for i in range(n):
        key = _make_embed_key(df.iloc[i])
        cache[key] = embeddings[i]
    
    save_embedding_cache(cache, keyed_path)
    print(f"  ✓ Migrated {len(cache):,} embeddings to keyed cache")
    return cache


def generate_embeddings(df: pd.DataFrame, use_cache: bool = False) -> tuple:
    """
    Generate sentence embeddings for all answers (incremental).
    
    Uses a keyed dict cache so that adding new agents only requires
    encoding their answers, not re-encoding everything.
    
    Args:
        df: DataFrame with AGENT, VIDEO, QUESTION_NUM, ANSWER columns.
        use_cache: If True, load/save keyed cache and only encode missing rows.
    """
    print("\n=== Generating Embeddings (Incremental) ===")
    
    keyed_path = config.EMBEDDING_CACHE["keyed"]
    legacy_path = config.EMBEDDING_CACHE["legacy_npy"]
    
    cache: dict = {}
    
    if use_cache:
        # Try keyed cache first
        if os.path.exists(keyed_path):
            cache = load_embedding_cache(keyed_path)
        # Fall back to legacy .npy and migrate
        elif os.path.exists(legacy_path):
            cache = migrate_npy_to_keyed_cache(legacy_path, keyed_path, df)
    
    # Identify rows that need embedding
    keys_needed = [_make_embed_key(row) for _, row in df.iterrows()]
    missing_indices = [i for i, k in enumerate(keys_needed) if k not in cache]
    
    print(f"  Total rows: {len(df):,}")
    print(f"  Cached:     {len(df) - len(missing_indices):,}")
    print(f"  To encode:  {len(missing_indices):,}")
    
    if missing_indices:
        # Load model only if there's work to do
        print("Loading sentence transformer model...")
        model = SentenceTransformer(
            EMBED_MODEL_SMALL,
            #model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": "bfloat16"},
            #tokenizer_kwargs={"padding_side": "left"},
        )
        print("✓ Model loaded")
        
        missing_answers = df.iloc[missing_indices]["ANSWER"].astype(str).tolist()
        
        print(f"⚠ Encoding {len(missing_answers)} answers... (this may take time)")
        
        CHECKPOINT = config.INCREMENTAL_CONFIG.get("embed_batch_checkpoint", 500)
        
        for start in range(0, len(missing_answers), CHECKPOINT):
            end = min(start + CHECKPOINT, len(missing_answers))
            batch_answers = missing_answers[start:end]
            batch_embeddings = model.encode(batch_answers, show_progress_bar=True, batch_size=BATCH_SIZE)
            
            # Store in cache
            for j, emb in enumerate(batch_embeddings):
                idx = missing_indices[start + j]
                key = keys_needed[idx]
                cache[key] = emb
            
            # Checkpoint save
            if use_cache:
                save_embedding_cache(cache, keyed_path)
                print(f"  ✓ Checkpoint: {len(cache):,} embeddings saved ({end}/{len(missing_answers)} new encoded)")
        
        print(f"✓ Encoded {len(missing_indices)} new answers")
    else:
        print("✓ All embeddings already cached — nothing to encode")
    
    # Build aligned array from cache in CSV row order
    embeddings = np.vstack([cache[k] for k in keys_needed])
    print(f"✓ Final embedding matrix: {embeddings.shape}")
    
    # Also save legacy .npy for backward compat (cheap — just a write)
    if use_cache:
        utils.ensure_output_dir(os.path.dirname(legacy_path))
        np.save(legacy_path, embeddings)
    
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
    
    # Add canonical video region columns
    df = utils.add_video_region_columns(df, video_col="VIDEO", id_col="VIDEO_NUM", region_col="VIDEO_REGION")
    df['VIDEO_SECTOR'] = df['VIDEO_REGION'].map({"lima": "Lima", "nyc": "NYC"}).fillna("Unknown")
    
    # Separate humans and VLMs (use dynamic detection)
    human_mask = df["AGENT"].str.startswith("human_")
    vlm_mask = ~human_mask
    
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
                        color=config.PLOT_COLORS["VLM"],
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
                        color=config.PLOT_COLORS["HUMAN_LIMA"],
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
                        color=config.PLOT_COLORS["HUMAN_NYC"],
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
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ LIMA HUMANS ━━━',
                                             markerfacecolor='blue', markersize=0, linestyle='None'))
                all_labels.append('━━━ LIMA HUMANS ━━━')
                for h in handles_lima:
                    all_handles.append(h)
                    all_labels.append(h.get_label())
            
            if handles_nyc:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ NYC HUMANS ━━━',
                                             markerfacecolor='darkblue', markersize=0, linestyle='None'))
                all_labels.append('━━━ NYC HUMANS ━━━')
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
        
        #get min,max of embeddings_2d_block for consistent axis limits
        min_x, max_x = embeddings_2d_block[:, 0].min(), embeddings_2d_block[:, 0].max()
        min_y, max_y = embeddings_2d_block[:, 1].min(), embeddings_2d_block[:, 1].max()    

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
                        color=config.PLOT_COLORS["VLM"],
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
                        color=config.PLOT_COLORS["HUMAN_LIMA"],
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
                        color=config.PLOT_COLORS["HUMAN_NYC"],
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
                                             markerfacecolor=config.PLOT_COLORS["VLM"], markersize=0, linestyle='None'))
                all_labels.append('━━━ VLMs ━━━')
                for h in handles_vlm:
                    all_handles.append(h)
                    all_labels.append(h.get_label())

            if handles_lima:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ LIMA HUMANS ━━━',
                                             markerfacecolor=config.PLOT_COLORS["HUMAN_LIMA"], markersize=0, linestyle='None'))
                all_labels.append('━━━ LIMA HUMANS ━━━')
                for h in handles_lima:
                    all_handles.append(h)
                    all_labels.append(h.get_label())

            if handles_nyc:
                all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ NYC HUMANS ━━━',
                                             markerfacecolor=config.PLOT_COLORS["HUMAN_NYC"], markersize=0, linestyle='None'))
                all_labels.append('━━━ NYC HUMANS ━━━')
                for h in handles_nyc:
                    all_handles.append(h)
                    all_labels.append(h.get_label())

            ax.set_title(f"PCA - Block {block} - Videos {sector} (Q{(block-1)*5+1}-Q{block*5})", 
                        fontsize=config.PLOT_CONFIG["title_fontsize"], fontweight="bold")
            ax.set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=11)
            ax.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=11)
            ax.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(1, 0.5), 
                     frameon=True, fontsize=config.PLOT_CONFIG["legend_fontsize"], ncol=1)
            ax.grid(True, alpha=0.3)
            
            # add a little padding to the limits for better visualization
            x_padding = (max_x - min_x) * 0.05
            y_padding = (max_y - min_y) * 0.05
            ax.set_xlim(min_x - x_padding, max_x + x_padding)
            ax.set_ylim(min_y - y_padding, max_y + y_padding)            
            
            plt.tight_layout()
            
            sector_label = sector.lower()
            output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, 
                                      f"pca_block{block}_{sector_label}.png")
            plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
            plt.close()
            
            print(f"  Saved: {os.path.basename(output_path)}")

def reduce_and_plot_combined_pca(embeddings, df: pd.DataFrame) -> None:
    """Create a combined PCA plot for both Lima and NYC regions."""
    print("\n=== PCA Combined Plot (Lima + NYC) ===")

    # Fit PCA with ALL videos (Lima + NYC)
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    variance = pca.explained_variance_ratio_

    # Plot with individual agent colors and markers
    fig, ax = plt.subplots(figsize=(14, 10))

    # Collect handles for grouped legend
    handles_vlm = []
    handles_lima = []
    handles_nyc = []

    # Plot VLMs (orange tones)
    for agent in config.VLM_AGENTS:
        if agent in df["AGENT"].values:
            agent_mask = df["AGENT"] == agent
            mask_array = agent_mask.values

            scatter = ax.scatter(
                embeddings_2d[mask_array, 0],
                embeddings_2d[mask_array, 1],
                color=config.PLOT_COLORS["VLM"],
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
        if agent in df["AGENT"].values:
            agent_mask = df["AGENT"] == agent
            mask_array = agent_mask.values
            scatter = ax.scatter(
                embeddings_2d[mask_array, 0],
                embeddings_2d[mask_array, 1],
                color=config.PLOT_COLORS["HUMAN_LIMA"],
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
        if agent in df["AGENT"].values:
            agent_mask = df["AGENT"] == agent
            mask_array = agent_mask.values
            scatter = ax.scatter(
                embeddings_2d[mask_array, 0],
                embeddings_2d[mask_array, 1],
                color=config.PLOT_COLORS["HUMAN_NYC"],
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
                                     markerfacecolor=config.PLOT_COLORS["VLM"], markersize=0, linestyle='None'))
        all_labels.append('━━━ VLMs ━━━')
        for h in handles_vlm:
            all_handles.append(h)
            all_labels.append(h.get_label())

    if handles_lima:
        all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ LIMA HUMANS ━━━',
                                     markerfacecolor=config.PLOT_COLORS["HUMAN_LIMA"], markersize=0, linestyle='None'))
        all_labels.append('━━━ LIMA HUMANS ━━━')
        for h in handles_lima:
            all_handles.append(h)
            all_labels.append(h.get_label())

    if handles_nyc:
        all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ NYC HUMANS ━━━',
                                     markerfacecolor=config.PLOT_COLORS["HUMAN_NYC"], markersize=0, linestyle='None'))
        all_labels.append('━━━ NYC HUMANS ━━━')
        for h in handles_nyc:
            all_handles.append(h)
            all_labels.append(h.get_label())

    ax.set_title(f"PCA - Combined Plot (Lima + NYC)", 
                fontsize=config.PLOT_CONFIG["title_fontsize"], fontweight="bold")
    ax.set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=11)
    ax.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(1, 0.5), 
             frameon=True, fontsize=config.PLOT_CONFIG["legend_fontsize"], ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, "pca_combined_lima_nyc.png")
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()

    print(f"  Saved: {os.path.basename(output_path)}")


def reduce_and_plot_combined_pca_by_block(embeddings, df: pd.DataFrame) -> None:
    """Create PCA plots combining Lima and NYC for each block."""
    print("\n=== PCA Combined Plot by Block (Lima + NYC) ===")

    for block in sorted(df["BLOCK"].unique()):
        # Fit PCA for the current block with ALL videos (Lima + NYC)
        block_mask = df["BLOCK"] == block
        embeddings_block = embeddings[block_mask]
        df_block = df[block_mask]

        print(f"Block {block}: Fitting PCA with {len(df_block)} total samples")

        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_block)
        variance = pca.explained_variance_ratio_

        # Plot with individual agent colors and markers
        fig, ax = plt.subplots(figsize=(14, 10))

        # Collect handles for grouped legend
        handles_vlm = []
        handles_lima = []
        handles_nyc = []

        # Plot VLMs (orange tones)
        for agent in config.VLM_AGENTS:
            if agent in df_block["AGENT"].values:
                agent_mask = df_block["AGENT"] == agent
                mask_array = agent_mask.values

                scatter = ax.scatter(
                    embeddings_2d[mask_array, 0],
                    embeddings_2d[mask_array, 1],
                    color=config.PLOT_COLORS["VLM"],
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
            if agent in df_block["AGENT"].values:
                agent_mask = df_block["AGENT"] == agent
                mask_array = agent_mask.values
                scatter = ax.scatter(
                    embeddings_2d[mask_array, 0],
                    embeddings_2d[mask_array, 1],
                    color=config.PLOT_COLORS["HUMAN_LIMA"],
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
            if agent in df_block["AGENT"].values:
                agent_mask = df_block["AGENT"] == agent
                mask_array = agent_mask.values
                scatter = ax.scatter(
                    embeddings_2d[mask_array, 0],
                    embeddings_2d[mask_array, 1],
                    color=config.PLOT_COLORS["HUMAN_NYC"],
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
                                         markerfacecolor=config.PLOT_COLORS["VLM"], markersize=0, linestyle='None'))
            all_labels.append('━━━ VLMs ━━━')
            for h in handles_vlm:
                all_handles.append(h)
                all_labels.append(h.get_label())

        if handles_lima:
            all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ LIMA HUMANS ━━━',
                                         markerfacecolor=config.PLOT_COLORS["HUMAN_LIMA"], markersize=0, linestyle='None'))
            all_labels.append('━━━ LIMA HUMANS ━━━')
            for h in handles_lima:
                all_handles.append(h)
                all_labels.append(h.get_label())

        if handles_nyc:
            all_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='━━━ NYC HUMANS ━━━',
                                         markerfacecolor=config.PLOT_COLORS["HUMAN_NYC"], markersize=0, linestyle='None'))
            all_labels.append('━━━ NYC HUMANS ━━━')
            for h in handles_nyc:
                all_handles.append(h)
                all_labels.append(h.get_label())

        ax.set_title(f"PCA - Block {block} Combined (Lima + NYC)", 
                    fontsize=config.PLOT_CONFIG["title_fontsize"], fontweight="bold")
        ax.set_xlabel(f"PC1 ({variance[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({variance[1]:.1%})", fontsize=11)
        ax.legend(all_handles, all_labels, loc="center left", bbox_to_anchor=(1, 0.5), 
                 frameon=True, fontsize=config.PLOT_CONFIG["legend_fontsize"], ncol=1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(config.OUTPUT_EMBEDDINGS_DIR, f"pca_block{block}_combined.png")
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

    # Generate combined PCA plots by block
    #reduce_and_plot_combined_pca_by_block(embeddings, df)

    print("\n" + "=" * 60)
    print("✓ Embedding analysis complete!")
    print(f"✓ Generated 20 plots (4 blocks x 2 sectors + 4 combined) in: {config.OUTPUT_EMBEDDINGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding analysis with UMAP and PCA")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache (regenerate)")
    parser.add_argument("--vlm-mode", choices=["first", "mean"], default="first", help="first or mean for VLM processing")
    args = parser.parse_args()
    
    main(use_cache=not args.no_cache, vlm_mode=args.vlm_mode)


