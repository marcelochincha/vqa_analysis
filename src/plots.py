
def plot_by_block(reduced_embeddings : np.array, df: pd.DataFrame , reductor : str) -> None:
    """Create PCA plots separated by block and video sector."""
    
    REQUIRED_COLUMNS = {"BLOCK", "VIDEO_SECTOR", "AGENT"}
    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {REQUIRED_COLUMNS}")
    
    
    print("\n=== PCA by Block and Sector ===")
    for block in sorted(df["BLOCK"].unique()):
        # Fit PCA once per block with ALL videos (Lima + NYC)
        block_mask = df["BLOCK"] == block
        embeddings_block = reduced_embeddings[block_mask]
        df_block = df[block_mask]
        min_x, max_x = embeddings_block[:, 0].min(), embeddings_block[:, 0].max()
        min_y, max_y = embeddings_block[:, 1].min(), embeddings_block[:, 1].max()    

        # Now plot each sector separately in the same embedding space
        for sector in ['Lima', 'NYC']: 
            sector_mask = df_block["VIDEO_SECTOR"] == sector
            embeddings_2d = embeddings_block[sector_mask]
            df_subset = df_block[sector_mask]
            print(f"  {sector}: {len(df_subset)} samples")
            if len(df_subset) < 1:
                print(f"Skipping (no data)")
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
            ax.set_xlabel(f"DIM 1", fontsize=11)
            ax.set_ylabel(f"DIM 2", fontsize=11)
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
                                      f"{reductor}_{block}_{sector_label}.png")
            plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
            plt.close()
            
            print(f"  Saved: {os.path.basename(output_path)}")
            

def plot_cosine_heatmap(df_agg,title,column, color_label, value_range, fmt,output_dir):
    REQUIRED_COLUMNS = {"AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR", column}
    if not REQUIRED_COLUMNS.issubset(df_agg.columns):
        raise ValueError(f"DataFrame must contain columns: {REQUIRED_COLUMNS}")
    
    regions = df_agg["VIDEO_SECTOR"].unique()
    blocks = df_agg["BLOCK"].unique()
    
    v_min = value_range[0]
    v_max = value_range[1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cmap", ["#ffffff", "#963fc9"])
    
    for region in regions:
        for block in blocks:
            print(f"Generating heatmap for {region} Block {block}...")
            subset = df_agg[(df_agg["VIDEO_SECTOR"] == region) & (df_agg["BLOCK"] == block)]
            if subset.empty:
                print(f"Skipping heatmap for {region} Block {block} (no data)")
                continue
            
            pivot_table = subset.pivot_table(
                index="AGENT_I", 
                columns="AGENT_J", 
                values=column
            )
            print(pivot_table)
            
            plot_similarity_heatmap(
                pivot_table,
                title= title +  f" - Block {block}|{region}",
                color_label= color_label,
                output_path= os.path.join(output_dir, f"heatmap_{region.lower()}_block{block}_{column}.png"),
                cmap=cmap,
                fmt = fmt,
                vmin=v_min,
                vmax=v_max
            )
            

def reduce_pca_by_block(embeddings, df):
    reduced_embeddings = np.zeros((embeddings.shape[0], 2))
    pca = PCA(n_components=2, random_state=42)
    for block in sorted(df["BLOCK"].unique()):
        block_mask = df["BLOCK"] == block
        embeddings_block = embeddings[block_mask]
        pca.fit(embeddings_block)
        print(f"Block {block}: Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
        reduced_embeddings[block_mask] = pca.transform(embeddings_block)
        
    return reduced_embeddings


from umap import UMAP
def reduce_umap_by_block(embeddings, df):
    reduced_embeddings = np.zeros((embeddings.shape[0], 2))
    for block in sorted(df["BLOCK"].unique()):
        block_mask = df["BLOCK"] == block
        embeddings_block = embeddings[block_mask]
        umap = UMAP(n_components=2, random_state=42)
        reduced_embeddings[block_mask] = umap.fit_transform(embeddings_block)
    return reduced_embeddings
