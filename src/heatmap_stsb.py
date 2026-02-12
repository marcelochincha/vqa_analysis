"""STSB-RoBERTa Score heatmap analysis with auto-computation."""
import os
import pandas as pd
from itertools import combinations
from sentence_transformers import CrossEncoder
from src import config, utils
from src.heatmap_common import create_similarity_matrix, plot_similarity_heatmap, plot_agreement_heatmap, create_agreement_matrix

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
    
    # Load existing cache
    cache_path = config.STSB_SCORES["stsb_cache_by_pairs"]
    stsb_cache = load_stsb_cache_by_pairs(cache_path)
    
    # Find missing pairs
    missing_pairs = [p for p in needed_text_pairs if p not in stsb_cache]
    
    # Handle self-comparisons
    self_pairs = [p for p in missing_pairs if p[0] == p[1]]
    for p in self_pairs:
        stsb_cache[p] = 1.0
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
    print(df.info())
    # Aggregate by block and agent pair
    aggregated = df.groupby(['BLOCK', 'AGENT_I', 'AGENT_J'])["BERT_SCORE"].agg([
        ('MEAN_SCORE', 'mean'),
        ('COUNT', 'count'),
        ('MIN_SCORE', 'min'),
        ('MAX_SCORE', 'max'),
        ('STD_SCORE', 'std'),
    ]).reset_index()
    
    return aggregated


def generate_stsb_heatmaps(aggregated_df: pd.DataFrame, pairwise_df : pd.DataFrame) -> None:
    """Generate similarity and agreement heatmaps for STSB-RoBERTa scores."""
    print("\n=== Generating STSB-RoBERTa Heatmaps ===")
    
    utils.ensure_output_dir(config.OUTPUT_STSB_PLOTS)
    
    for block in range(1, config.NUM_BLOCKS + 1):
        matrix = create_similarity_matrix(aggregated_df, block, score_column="MEAN_SCORE")
        # Similarity heatmap
        similarity_path = os.path.join(
            config.OUTPUT_STSB_PLOTS,
            f"stsb_similarity_block{block}.png"
        )
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("cmap",["#ffffff", "#ebb540"])
        
        plot_similarity_heatmap(
            matrix,
            title=f"STSB-RoBERTa Similarity - Block {block}",
            output_path=similarity_path,
            cmap=cmap,  # Different colormap for STSB
            vmin=0.0,
            vmax=1.0,
            use_group_colors=True
        )
        
        # Agreement heatmap
        agreement_path = os.path.join(
            config.OUTPUT_STSB_PLOTS,
            f"stsb_agreement_block{block}.png"
        )
        
        #This is a little differnet since i dont need the matrix i need the pairwise scores to compute agreement percentages based on thresholds
        aggrement_matrix = create_agreement_matrix(pairwise_df, block, score_column="BERT_SCORE", threshold=0.5) # Using 0.5 as agreement threshold for STSB
        plot_agreement_heatmap(
            aggrement_matrix,
            title=f"STSB-RoBERTa Agreement - Block {block}",
            output_path=agreement_path,
            cmap=cmap,
            use_group_colors=True
        )
    
    print(f"✓ Generated {config.NUM_BLOCKS * 2} STSB-RoBERTa heatmap plots")


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
    
    # Generate heatmaps
    generate_stsb_heatmaps(aggregated_df, pairwise_df)
    
    print("\n" + "=" * 60)
    print("✓ STSB-RoBERTa analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
