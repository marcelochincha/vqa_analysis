"""STSB-RoBERTa Score heatmap analysis with auto-computation."""
import os
import pandas as pd
from itertools import combinations
from sentence_transformers import CrossEncoder
from src import config, utils
from src.heatmap_common import create_similarity_matrix, plot_similarity_heatmap, create_agreement_matrix

def load_stsb_model():
    """Load STSB-RoBERTa cross-encoder model."""
    print("Loading STSB-RoBERTa cross-encoder model...")
    try:
        model = CrossEncoder('cross-encoder/stsb-roberta-large')
        print("✓ STSB-RoBERTa model loaded")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


# ============================================================================
# Text-pair cache (like SMATCH pattern)
# ============================================================================

def load_stsb_cache_by_pairs(cache_path: str) -> dict:
    """Load STSB scores indexed by unique text pairs.
    
    Returns dict with sorted (text_a, text_b) tuple keys -> score.
    """
    if not os.path.exists(cache_path):
        return {}
    df = pd.read_csv(cache_path)
    cache = {}
    for _, row in df.iterrows():
        text_a = str(row["TEXT_A"]) if pd.notna(row["TEXT_A"]) else ""
        text_b = str(row["TEXT_B"]) if pd.notna(row["TEXT_B"]) else ""
        score = float(row["STSB_SCORE"]) if pd.notna(row["STSB_SCORE"]) else 0.0
        pair_key = tuple(sorted([text_a, text_b]))
        cache[pair_key] = score
    print(f"✓ Loaded {len(cache):,} unique STSB text-pair scores from cache")
    return cache


def save_stsb_cache_by_pairs(cache: dict, cache_path: str) -> None:
    """Save STSB text-pair cache to CSV."""
    rows = [{"TEXT_A": k[0], "TEXT_B": k[1], "STSB_SCORE": v} for k, v in cache.items()]
    df = pd.DataFrame(rows)
    utils.ensure_output_dir(os.path.dirname(cache_path))
    df.to_csv(cache_path, index=False)


def migrate_legacy_pairwise_to_text_cache(legacy_pairwise_path: str, text_cache_path: str, df: pd.DataFrame) -> dict:
    """Migrate legacy agent-based pairwise scores to text-pair cache.
    
    Reads the existing pairwise_stsb_scores.csv (agent pairs), matches with
    the current CSV to get answer texts, and builds the text-pair cache.
    This avoids re-computing scores that already exist.
    """
    print("\n--- Migrating legacy STSB pairwise scores to text-pair cache ---")
    
    if not os.path.exists(legacy_pairwise_path):
        print("  No legacy pairwise file found")
        return {}
    
    legacy_df = pd.read_csv(legacy_pairwise_path)
    print(f"  Found {len(legacy_df):,} agent-pair scores in legacy file")
    
    # Normalize texts in current CSV
    df["ANSWER_NORMALIZED"] = df["ANSWER"].apply(utils.normalize_text)
    
    # Build lookup: (agent, video, qnum) -> normalized text
    answer_lookup = {}
    for _, row in df.iterrows():
        key = (row["AGENT"], row["VIDEO"], row["QUESTION_NUM"])
        answer_lookup[key] = row["ANSWER_NORMALIZED"]
    
    # Convert agent pairs to text pairs
    text_cache = {}
    migrated = 0
    skipped = 0
    
    for _, row in legacy_df.iterrows():
        agent_i = row["AGENT_I"]
        agent_j = row["AGENT_J"]
        video = row["VIDEO"]
        qnum = row["QUESTION_NUM"]
        score = float(row[""])  # Legacy used BERT_SCORE column name
        
        # Look up the texts for this agent pair
        key_i = (agent_i, video, qnum)
        key_j = (agent_j, video, qnum)
        
        text_i = answer_lookup.get(key_i)
        text_j = answer_lookup.get(key_j)
        
        if text_i is None or text_j is None:
            skipped += 1
            continue
        
        # Store with sorted tuple to handle (a,b) vs (b,a)
        text_pair = tuple(sorted([text_i, text_j]))
        
        # Only store if not already present (first score wins for duplicates)
        if text_pair not in text_cache:
            text_cache[text_pair] = score
            migrated += 1
    
    print(f"  Migrated {migrated:,} unique text-pair scores")
    print(f"  Skipped {skipped:,} rows (agents not in current CSV)")
    
    dedup_saved = len(legacy_df) - len(text_cache)
    if dedup_saved > 0:
        print(f"  Deduplication saved {dedup_saved:,} redundant pairs ({dedup_saved/len(legacy_df)*100:.1f}%)")
    
    # Save migrated cache
    save_stsb_cache_by_pairs(text_cache, text_cache_path)
    print(f"  ✓ Saved text-pair cache: {os.path.basename(text_cache_path)}")
    
    return text_cache


def compute_stsb_scores() -> pd.DataFrame:
    """Compute pairwise STSB-RoBERTa scores with text-pair deduplication and checkpointing.
    
    Three-phase approach (mirrors SMATCH):
      Phase 1: Identify unique text pairs needed.
      Phase 2: Score only missing text pairs (with periodic checkpointing).
      Phase 3: Expand to all agent pairs.
    """
    print("\n=== Computing STSB-RoBERTa Scores (Incremental) ===")
    
    # Load answers and discover agents
    df = utils.load_answers()
    agents = utils.get_agent_groups()["all"]
    
    # Normalize texts for deduplication
    df["ANSWER_NORMALIZED"] = df["ANSWER"].apply(utils.normalize_text)
    
    total_answers = len(df)
    unique_texts = df["ANSWER_NORMALIZED"].nunique()
    print(f"  Total answers: {total_answers:,}")
    print(f"  Unique texts:  {unique_texts:,}")
    
    # Build mapping: (agent, video, qnum) -> normalized text
    answer_text_map = {}
    for _, row in df.iterrows():
        key = (row["AGENT"], row["VIDEO"], row["QUESTION_NUM"])
        answer_text_map[key] = row["ANSWER_NORMALIZED"]
    
    # Identify all unique text pairs needed
    needed_text_pairs = set()
    agent_pair_to_text_pair = {}
    total_agent_pairs = 0
    
    for (video, question_num), group in df.groupby(["VIDEO", "QUESTION_NUM"]):
        available_agents = set(group["AGENT"].unique())
        for agent_i, agent_j in combinations(agents, 2):
            if agent_i not in available_agents or agent_j not in available_agents:
                continue
            total_agent_pairs += 1
            key_i = (agent_i, video, question_num)
            key_j = (agent_j, video, question_num)
            text_i = answer_text_map.get(key_i, "")
            text_j = answer_text_map.get(key_j, "")
            text_pair = tuple(sorted([text_i, text_j]))
            needed_text_pairs.add(text_pair)
            agent_pair_key = (video, question_num, agent_i, agent_j)
            agent_pair_to_text_pair[agent_pair_key] = (text_i, text_j)
    
    # Load or migrate cache
    cache_path = config.STSB_SCORES["stsb_cache_by_pairs"]
    legacy_path = config.STSB_SCORES["pairwise"]
    
    # Check if we need to migrate from legacy
    if not os.path.exists(cache_path) and os.path.exists(legacy_path):
        print("\n" + "=" * 60)
        print("LEGACY PAIRWISE FILE DETECTED")
        print("=" * 60)
        print("\nMigrating existing pairwise_stsb_scores.csv to text-pair cache...")
        print("This will deduplicate scores and speed up future runs.")
        stsb_cache = migrate_legacy_pairwise_to_text_cache(legacy_path, cache_path, df)
        print("\n✓ Migration complete!\n")
    else:
        stsb_cache = load_stsb_cache_by_pairs(cache_path)
    
    # Find missing pairs
    missing_pairs = [p for p in needed_text_pairs if p not in stsb_cache]
    
    # Handle self-comparisons
    self_pairs = [p for p in missing_pairs if p[0] == p[1]]
    for p in self_pairs:
        stsb_cache[p] = 1.0 #stbs should always be 1 for identical pairs but we can set it here to avoid unnecessary model calls
    missing_pairs = [p for p in missing_pairs if p[0] != p[1]]
    
    dedup_pct = (1 - len(needed_text_pairs) / total_agent_pairs) * 100 if total_agent_pairs else 0
    print(f"  Total agent pairs:     {total_agent_pairs:,}")
    print(f"  Unique text pairs:     {len(needed_text_pairs):,} (dedup saved {dedup_pct:.1f}%)")
    print(f"  Already cached:        {len(needed_text_pairs) - len(missing_pairs) - len(self_pairs):,}")
    print(f"  To compute:            {len(missing_pairs):,}")
    
    if missing_pairs:
        model = load_stsb_model()
        
        CHECKPOINT = config.INCREMENTAL_CONFIG.get("stsb_pair_checkpoint", 1000)
        batch_size = 32
        
        for chunk_start in range(0, len(missing_pairs), CHECKPOINT):
            chunk_end = min(chunk_start + CHECKPOINT, len(missing_pairs))
            chunk = missing_pairs[chunk_start:chunk_end]
            
            # Prepare input pairs for the model
            input_pairs = [[p[0], p[1]] for p in chunk]
            
            # Score in sub-batches
            all_scores = []
            for i in range(0, len(input_pairs), batch_size):
                batch = input_pairs[i:i + batch_size]
                scores = model.predict(batch)
                all_scores.extend(scores)
            
            # Store in cache
            for pair, score in zip(chunk, all_scores):
                stsb_cache[pair] = float(score)
            
            # Checkpoint save
            save_stsb_cache_by_pairs(stsb_cache, cache_path)
            print(f"  ✓ Checkpoint: {len(stsb_cache):,} text pairs cached ({chunk_end}/{len(missing_pairs)} new scored)")
        
        print(f"✓ Scored {len(missing_pairs)} new text pairs")
    else:
        print("✓ All text pairs already cached — no model loading needed")
    
    # Phase 3: Expand to agent pairs
    print("\n--- Expanding to agent pairs ---")
    results = []
    print(f"  Total agent pairs to process: {len(agent_pair_to_text_pair):,}")
    for (video, question_num, agent_i, agent_j), (text_i, text_j) in agent_pair_to_text_pair.items():
        text_pair = tuple(sorted([text_i, text_j]))
        score = stsb_cache.get(text_pair, 0.0)
        results.append({
            "VIDEO": video,
            "QUESTION_NUM": question_num,
            "AGENT_I": agent_i,
            "AGENT_J": agent_j,
            "BERT_SCORE": float(score),
        })
    
    pairwise_df = pd.DataFrame(results)
    
    utils.ensure_output_dir(config.OUTPUT_STSB_DIR)
    pairwise_df.to_csv(config.STSB_SCORES["pairwise"], index=False)
    print(f"✓ Saved {len(pairwise_df):,} pairwise scores: {config.STSB_SCORES['pairwise']}")
    
    return pairwise_df

def aggregate_stsb_scores(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    df = pairwise_df.copy()
        
    # Add block column
    utils.compute_blocks(df)
    #print(df.head())

    aggregated = df.groupby(['BLOCK', 'AGENT_I', 'AGENT_J'])["BERT_SCORE"].agg([
        ('MEAN_SCORE', 'mean'),
        ('COUNT', 'count'),
        ('MIN_SCORE', 'min'),
        ('MAX_SCORE', 'max'),
        ('STD_SCORE', 'std'),
    ]).reset_index()
    
    #show the block 2 mean scores
    print("\nSample of aggregated scores (Block 2):")
    return aggregated


def generate_stsb_heatmaps(aggregated_df: pd.DataFrame, pairwise_df : pd.DataFrame) -> None:
    """Generate similarity and agreement heatmaps for STSB-RoBERTa scores."""
    print("\n=== Generating STSB-RoBERTa Heatmaps ===")
    
    utils.ensure_output_dir(config.OUTPUT_STSB_PLOTS)
    
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cmap", ["#ffffff", "#ebb540"])
    region_labels = {"both": "All", "lima": "Lima", "nyc": "NYC"}
    total_plots = 0

    for block in range(1, config.NUM_BLOCKS + 1):
        for region in ["both", "lima", "nyc"]:
            matrix = create_similarity_matrix(
                aggregated_df,
                block,
                score_column="BERT_SCORE",
                video_region=region,
                pairwise_df=pairwise_df,
            )

            if region == "both":
                similarity_path = os.path.join(config.OUTPUT_STSB_PLOTS, f"stsb_similarity_block{block}.png")
            else:
                similarity_path = os.path.join(config.OUTPUT_STSB_PLOTS, f"stsb_similarity_block{block}_{region}.png")

            region_title = "" if region == "both" else f" ({region_labels[region]})"
            plot_similarity_heatmap(
                matrix,
                title=f"STSB-RoBERTa Similarity - Block {block}{region_title}",
                color_label="Similarity Score",
                output_path=similarity_path,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                use_group_colors=True
            )
            if matrix is not None and not matrix.empty:
                total_plots += 1

            if region == "both":
                agreement_path = os.path.join(config.OUTPUT_STSB_PLOTS, f"stsb_agreement_block{block}.png")
            else:
                agreement_path = os.path.join(config.OUTPUT_STSB_PLOTS, f"stsb_agreement_block{block}_{region}.png")

            aggrement_matrix = create_agreement_matrix(
                pairwise_df,
                block,
                score_column="BERT_SCORE",
                threshold=0.5,
                video_region=region,
            )
            plot_similarity_heatmap(
                aggrement_matrix,
                title=f"STSB-RoBERTa Agreement - Block {block}{region_title}",
                color_label="Agreement (%)",
                output_path=agreement_path,
                cmap=cmap,
                vmin=0,
                vmax=100,
                fmt="%d",
                use_group_colors=True
            )
            if aggrement_matrix is not None and not aggrement_matrix.empty:
                total_plots += 1
    
    print(f"✓ Generated {total_plots} STSB-RoBERTa heatmap plots")


def main():
    """Main execution: compute scores if needed, then generate heatmaps.
    
    Supports incremental updates: if a pairwise file exists but new agents
    are detected in the input CSV, re-runs compute_stsb_scores() which will
    leverage the text-pair cache (only scoring new unique text pairs).
    """
    print("=" * 60)
    print("STSB-RoBERTa SCORE HEATMAP ANALYSIS")
    print("=" * 60)
    
    # Check if scores exist
    pairwise_exists = os.path.exists(config.STSB_SCORES["pairwise"])
    aggregated_exists = os.path.exists(config.STSB_SCORES["aggregated"])
    
    # Detect new agents
    new_agents = set()
    if pairwise_exists:
        new_agents = utils.detect_new_agents(config.STSB_SCORES["pairwise"])
    
    needs_recompute = not pairwise_exists or len(new_agents) > 0
    
    if needs_recompute:
        if new_agents:
            print(f"\n⚡ New agents detected: {sorted(new_agents)}")
            print("  Will recompute pairwise scores (text-pair cache will be reused)")
        else:
            print("\n⚠ Pairwise STSB-RoBERTa scores not found. Computing...")
        
        pairwise_df = compute_stsb_scores()
        aggregated_df = aggregate_stsb_scores(pairwise_df)
    elif not aggregated_exists:
        print("\n⚠ Aggregated STSB-RoBERTa scores not found. Aggregating...")
        pairwise_df, _ = utils.load_metric_scores(config.STSB_SCORES["pairwise"])
        aggregated_df = aggregate_stsb_scores(pairwise_df)
    else:
        print("\n✓ STSB-RoBERTa scores already computed. Loading...")
        pairwise_df, aggregated_df = utils.load_metric_scores(
            config.STSB_SCORES["pairwise"],
            config.STSB_SCORES["aggregated"]
        )
    
    # #print some of the pairwise scores to verify the score showing the original TEXT_A and TEXT_B from the cache to verify the migration worked
    # print("\nSample of pairwise scores with original texts (from cache):")
    # #join the pairwise_df with the original answers
    # df_answers = utils.load_answers()
    # #drop duplicates of vlms 
    # df_answers = df_answers[["AGENT", "VIDEO", "QUESTION_NUM", "ANSWER"]].drop_duplicates()
    
    #     # --- traer ANSWER_I ---
    # df = pairwise_df.merge(
    #     df_answers[["VIDEO", "QUESTION_NUM", "AGENT", "ANSWER"]],
    #     left_on=["VIDEO", "QUESTION_NUM", "AGENT_I"],
    #     right_on=["VIDEO", "QUESTION_NUM", "AGENT"],
    #     how="left"
    # )

    # df = df.rename(columns={"ANSWER": "ANSWER_I"})
    # df = df.drop(columns=["AGENT"])

    # # --- traer ANSWER_J ---
    # df = df.merge(
    #     df_answers[["VIDEO", "QUESTION_NUM", "AGENT", "ANSWER"]],
    #     left_on=["VIDEO", "QUESTION_NUM", "AGENT_J"],
    #     right_on=["VIDEO", "QUESTION_NUM", "AGENT"],
    #     how="left"
    # )

    # df = df.rename(columns={"ANSWER": "ANSWER_J"})
    # df = df.drop(columns=["AGENT"])
    
    
    # print("\nSample of pairwise scores with texts:")
    # sample_rows = df.head(10)
    # for _, row in sample_rows.iterrows():
    #     print(f"VIDEO: {row['VIDEO']}, QNUM: {row['QUESTION_NUM']}, AGENT_I: {row['AGENT_I']}, AGENT_J: {row['AGENT_J']}, BERT_SCORE: {row['BERT_SCORE']}")
    #     print(f"  TEXT_A: {row['ANSWER_I']}")
    #     print(f"  TEXT_B: {row['ANSWER_J']}")
    #     print()
        
    # #show samples with highest and lowest scores
    # print("\nSample of highest pairwise scores:")
    # sample_high = df[(df["QUESTION_NUM"] >= 6) & (df["QUESTION_NUM"] <= 10)].sort_values(by="BERT_SCORE", ascending=False).head(5)
    # print(sample_high)
    
    # Generate heatmaps
    generate_stsb_heatmaps(aggregated_df, pairwise_df)
    
    print("\n" + "=" * 60)
    print("✓ STSB-RoBERTa analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
