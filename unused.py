#SMATCH  COMPUTE
def compute_smatch_scores_legacy() -> pd.DataFrame:
    """Legacy method: Compute pairwise SMATCH scores without text deduplication.
    
    This function operates in two phases:
    1. Parse all answers to AMR graphs (GPU, batched, with checkpointing)
    2. Compute SMATCH scores for all agent pairs (CPU, threaded, with checkpointing)
    
    Returns
    -------
    pd.DataFrame
        Pairwise scores with VIDEO, QUESTION_NUM, AGENT_I, AGENT_J, score columns.
    """
    print("\n=== Computing SMATCH Scores (Legacy Method) ===")
    
    # Load AMR parser
    stog = load_amr_parser()
    
    # Load answers
    df = utils.load_answers()
    agents = utils.get_agent_groups()["all"]
    
    # ========== PHASE 1: Parse answers to AMR graphs (with caching) ==========
    print("\n--- Phase 1: Parsing answers to AMR graphs ---")
    
    # Load existing AMR cache
    amr_cache = load_amr_cache(config.SMATCH_SCORES["amr_cache"])
    initial_cache_size = len(amr_cache)
    
    # Identify missing AMR parses
    missing_items = []
    for idx, row in df.iterrows():
        key = (row["AGENT"], row["VIDEO"], row["QUESTION_NUM"])
        if key not in amr_cache:
            missing_items.append({
                'key': key,
                'answer': str(row["ANSWER"])
            })
    
    total_needed = len(df)
    total_missing = len(missing_items)
    print(f"  Found {initial_cache_size}/{total_needed} AMR graphs in cache")
    print(f"  Need to parse {total_missing} new answers")
    
    # Parse missing answers in batches
    if total_missing > 0:
        BATCH_SIZE = config.SMATCH_CONFIG["batch_size_amr"]
        CHECKPOINT_FREQ = config.SMATCH_CONFIG["checkpoint_amr"]
        
        for i in range(0, total_missing, BATCH_SIZE):
            batch_items = missing_items[i:i+BATCH_SIZE]
            batch_texts = [item['answer'] for item in batch_items]
            
            # Parse batch (single GPU call)
            try:
                batch_graphs = stog.parse_sents(batch_texts)
                
                # Update cache with results
                for item, graph in zip(batch_items, batch_graphs):
                    amr_cache[item['key']] = graph if graph else ""
                
            except Exception as e:
                print(f"  Warning: Batch parsing failed: {e}")
                # Add empty graphs for failed parses
                for item in batch_items:
                    amr_cache[item['key']] = ""
            
            # Progress update
            parsed_so_far = min(i + BATCH_SIZE, total_missing)
            print(f"  Parsed {initial_cache_size + parsed_so_far}/{total_needed} answers ({parsed_so_far}/{total_missing} new)")
            
            # Checkpoint: save cache periodically
            if (i + BATCH_SIZE) % CHECKPOINT_FREQ == 0 or (i + BATCH_SIZE) >= total_missing:
                save_amr_cache(amr_cache, config.SMATCH_SCORES["amr_cache"])
                print(f"  ✓ Checkpoint: AMR cache saved ({len(amr_cache)} graphs)")
        
        print(f"✓ Phase 1 complete: {len(amr_cache)} total AMR graphs in cache")
    else:
        print(f"✓ Phase 1 complete: All AMR graphs already cached")
    
    # ========== PHASE 2: Compute SMATCH scores (with threading) ==========
    print("\n--- Phase 2: Computing SMATCH scores for agent pairs ---")
    
    # Load existing partial scores
    partial_path = config.SMATCH_SCORES["pairwise_partial"]
    existing_scores_df = load_scores_partial(partial_path)
    
    # Create set of already computed pairs for fast lookup
    existing_pairs = set()
    if not existing_scores_df.empty:
        for _, row in existing_scores_df.iterrows():
            pair_key = (row['VIDEO'], row['QUESTION_NUM'], row['AGENT_I'], row['AGENT_J'])
            existing_pairs.add(pair_key)
    
    # Identify missing score comparisons
    missing_pairs_info = []
    total_pairs_needed = 0
    
    for (video, question_num), group in df.groupby(["VIDEO", "QUESTION_NUM"]):
        available_agents = group["AGENT"].unique()
        
        # Check all agent pairs for this video/question
        for agent_i, agent_j in combinations(agents, 2):
            if agent_i not in available_agents or agent_j not in available_agents:
                continue
            
            total_pairs_needed += 1
            pair_key = (video, question_num, agent_i, agent_j)
            
            if pair_key not in existing_pairs:
                key_i = (agent_i, video, question_num)
                key_j = (agent_j, video, question_num)
                
                missing_pairs_info.append({
                    'video': video,
                    'qnum': question_num,
                    'agent_i': agent_i,
                    'agent_j': agent_j,
                    'key_i': key_i,
                    'key_j': key_j
                })
    
    total_missing_pairs = len(missing_pairs_info)
    print(f"  Found {len(existing_pairs)}/{total_pairs_needed} scores in checkpoint")
    print(f"  Need to compute {total_missing_pairs} new comparisons")
    
    # Compute missing pairs with threading
    if total_missing_pairs > 0:
        MAX_WORKERS = config.SMATCH_CONFIG["max_workers"]
        CHECKPOINT_FREQ = config.SMATCH_CONFIG["checkpoint_scores"]
        
        results_buffer = []
        completed = 0
        
        # Clear partial file and write existing scores back (if any)
        if not existing_scores_df.empty:
            existing_scores_df.to_csv(partial_path, mode='w', header=True, index=False)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all comparison tasks
            futures = {}
            for pair_info in missing_pairs_info:
                amr_i = amr_cache.get(pair_info['key_i'], "")
                amr_j = amr_cache.get(pair_info['key_j'], "")
                
                future = executor.submit(
                    compute_smatch_score,
                    amr_i,
                    amr_j
                )
                futures[future] = pair_info
            
            # Collect results as they complete
            for future in as_completed(futures):
                pair_info = futures[future]
                try:
                    score = future.result()
                    results_buffer.append({
                        "VIDEO": pair_info['video'],
                        "QUESTION_NUM": pair_info['qnum'],
                        "AGENT_I": pair_info['agent_i'],
                        "AGENT_J": pair_info['agent_j'],
                        "score": score
                    })
                except Exception as e:
                    print(f"  Warning: SMATCH comparison failed for {pair_info['agent_i']}-{pair_info['agent_j']}: {e}")
                    results_buffer.append({
                        "VIDEO": pair_info['video'],
                        "QUESTION_NUM": pair_info['qnum'],
                        "AGENT_I": pair_info['agent_i'],
                        "AGENT_J": pair_info['agent_j'],
                        "score": 0.0
                    })
                
                completed += 1
                
                # Progress update
                if completed % 100 == 0:
                    print(f"  Processed {len(existing_pairs) + completed}/{total_pairs_needed} comparisons ({completed}/{total_missing_pairs} new)")
                
                # Checkpoint: append to partial file
                if completed % CHECKPOINT_FREQ == 0:
                    save_scores_partial(results_buffer, partial_path, mode='a')
                    results_buffer = []
                    print(f"  ✓ Checkpoint: {len(existing_pairs) + completed} total scores saved")
        
        # Save remaining results in buffer
        if results_buffer:
            save_scores_partial(results_buffer, partial_path, mode='a')
        
        print(f"✓ Phase 2 complete: {total_pairs_needed} total SMATCH scores computed")
    else:
        print(f"✓ Phase 2 complete: All SMATCH scores already computed")
    
    # Load all scores (existing + new) and finalize
    final_scores_df = pd.read_csv(partial_path)
    
    # Save to final location
    utils.ensure_output_dir(config.OUTPUT_SMATCH_DIR)
    final_scores_df.to_csv(config.SMATCH_SCORES["pairwise"], index=False)
    print(f"✓ Saved: {config.SMATCH_SCORES['pairwise']}")
    
    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"✓ Cleaned up checkpoint file")
    
    return final_scores_df

# BIAS ANALYSIS
def plot_bias_distributions(bias_df: pd.DataFrame, region: str, output_dir: str) -> None:
    """
    Plot distribution charts to show bias patterns not visible in heatmaps.
    
    Creates:
    1. Violin plot: Distribution of bias signs per VLM across all Q6-Q10
    2. Bar plot: Average bias per VLM per question
    """
    print(f"\n=== Plotting Bias Distributions ({region.upper()}) ===")
    
    # 1. Violin plot: Overall distribution per VLM
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Keep consistent VLM order (alphabetical)
    vlm_order = sorted(bias_df["AGENT"].unique())
    
    sns.violinplot(
        data=bias_df,
        x="avg_bias_sign",
        y="AGENT",
        order=vlm_order,
        palette="Set2",
        ax=ax
    )
    
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    region_label = region.replace("_H_x_", " Humans × ").replace("_V", " Videos").replace("lima", "Lima").replace("nyc", "NYC")
    ax.set_title(
        f"VLM Bias Distribution: {region_label} (Q6-Q10)\\n"
        f"Unit Consensus Method",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Avg Bias Sign (-1 to +1)", fontsize=12)
    ax.set_ylabel("VLM Agent", fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"bias_distribution_{region}_overall.png")
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")
    
    # 2. Bar plot: Average bias per VLM per question
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Compute mean bias per VLM per question
    mean_bias = bias_df.groupby(["AGENT", "QUESTION_NUM"])["avg_bias_sign"].mean().reset_index()
    mean_bias_pivot = mean_bias.pivot(index="AGENT", columns="QUESTION_NUM", values="avg_bias_sign")
    #Keep consistent VLM order (alphabetical)
    vlm_order = sorted(mean_bias_pivot.index)
    vlm_order = mean_bias_pivot.mean(axis=1).sort_values().index
    mean_bias_pivot = mean_bias_pivot.loc[vlm_order]
    
    # Plot grouped bar chart
    x = np.arange(len(mean_bias_pivot))
    width = 0.15
    questions = sorted(mean_bias["QUESTION_NUM"].unique())
    
    for i, q in enumerate(questions):
        offset = width * (i - len(questions)/2 + 0.5)
        values = mean_bias_pivot[q].values
        ax.barh(x + offset, values, width, label=f"Q{q}")
    
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(mean_bias_pivot.index)
    ax.set_xlabel("Avg Bias Sign", fontsize=12)
    ax.set_ylabel("VLM Agent", fontsize=12)
    region_label = region.replace("_H_x_", " Humans × ").replace("_V", " Videos").replace("lima", "Lima").replace("nyc", "NYC")
    ax.set_title(
        f"Average Bias per Question: {region_label}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.legend(title="Question", loc="best")
    ax.set_xlim(-1.1, 1.1)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"bias_distribution_{region}_per_question.png")
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")
    
    
#EMBED ANALYSIS

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
            color=color,
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