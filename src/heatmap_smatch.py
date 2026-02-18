"""SMATCH Score heatmap analysis with AMR parsing and auto-computation."""
import os
import re
import pandas as pd
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import amrlib
from src import config, utils
from src.heatmap_common import create_similarity_matrix, plot_similarity_heatmap, create_agreement_matrix
import tqdm

def save_amr_cache(cache_dict: dict, cache_path: str) -> None:
    """Save AMR cache dictionary to CSV file.
    
    Parameters
    ----------
    cache_dict : dict
        Dictionary with (agent, video, qnum) tuple keys and AMR graph string values.
    cache_path : str
        Path to save the cache CSV file.
    """
    rows = []
    for (agent, video, qnum), amr_graph in cache_dict.items():
        rows.append({
            'AGENT': agent,
            'VIDEO': video,
            'QUESTION_NUM': qnum,
            'AMR_GRAPH': amr_graph
        })
    
    df = pd.DataFrame(rows)
    utils.ensure_output_dir(os.path.dirname(cache_path))
    df.to_csv(cache_path, index=False)


def load_amr_cache(cache_path: str) -> dict:
    """Load AMR cache from CSV file.
    
    Parameters
    ----------
    cache_path : str
        Path to the cache CSV file.
    
    Returns
    -------
    dict
        Dictionary with (agent, video, qnum) tuple keys and AMR graph string values.
    """
    if not os.path.exists(cache_path):
        return {}
    
    df = pd.read_csv(cache_path)
    cache_dict = {}
    
    for _, row in df.iterrows():
        key = (row['AGENT'], row['VIDEO'], row['QUESTION_NUM'])
        cache_dict[key] = str(row['AMR_GRAPH']) if pd.notna(row['AMR_GRAPH']) else ""
    
    print(f"✓ Loaded {len(cache_dict)} existing AMR graphs from cache")
    return cache_dict


def save_scores_partial(results_list: list, partial_path: str, mode: str = 'a') -> None:
    """Save partial SMATCH scores to CSV file.
    
    Parameters
    ----------
    results_list : list
        List of score dictionaries with VIDEO, QUESTION_NUM, AGENT_I, AGENT_J, score.
    partial_path : str
        Path to save the partial scores CSV file.
    mode : str, optional
        Write mode: 'w' for new file, 'a' for append (default: 'a').
    """
    if not results_list:
        return
    
    df = pd.DataFrame(results_list)
    utils.ensure_output_dir(os.path.dirname(partial_path))
    
    # Write header only if file doesn't exist or mode is 'w'
    write_header = mode == 'w' or not os.path.exists(partial_path)
    df.to_csv(partial_path, mode=mode, header=write_header, index=False)


def load_scores_partial(partial_path: str) -> pd.DataFrame:
    """Load partial SMATCH scores from CSV file.
    
    Parameters
    ----------
    partial_path : str
        Path to the partial scores CSV file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with existing scores, or empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(partial_path):
        return pd.DataFrame()
    
    df = pd.read_csv(partial_path)
    print(f"✓ Loaded {len(df)} existing SMATCH scores from checkpoint")
    return df


# ============================================================================
# TEXT-BASED DEDUPLICATION FUNCTIONS (OPTIMIZED)
# ============================================================================

def normalize_answer_text(text: str) -> str:
    """Normalize answer text for deduplication matching.
    
    Delegates to the shared utils.normalize_text() function.
    """
    return utils.normalize_text(text)


def save_amr_cache_by_text(text_to_amr_dict: dict, cache_path: str) -> None:
    """Save AMR cache indexed by unique answer texts.
    
    Parameters
    ----------
    text_to_amr_dict : dict
        Dictionary with normalized text keys and AMR graph string values.
    cache_path : str
        Path to save the cache CSV file.
    """
    rows = []
    for text, amr_graph in text_to_amr_dict.items():
        rows.append({
            'ANSWER_TEXT': text,
            'AMR_GRAPH': amr_graph
        })
    
    df = pd.DataFrame(rows)
    utils.ensure_output_dir(os.path.dirname(cache_path))
    df.to_csv(cache_path, index=False)


def load_amr_cache_by_text(cache_path: str) -> dict:
    """Load AMR cache indexed by unique answer texts.
    
    Parameters
    ----------
    cache_path : str
        Path to the cache CSV file.
    
    Returns
    -------
    dict
        Dictionary with normalized text keys and AMR graph string values.
    """
    if not os.path.exists(cache_path):
        return {}
    
    df = pd.read_csv(cache_path)
    cache_dict = {}
    
    for _, row in df.iterrows():
        text = str(row['ANSWER_TEXT']) if pd.notna(row['ANSWER_TEXT']) else ""
        amr = str(row['AMR_GRAPH']) if pd.notna(row['AMR_GRAPH']) else ""
        cache_dict[text] = amr
    
    print(f"✓ Loaded {len(cache_dict)} unique AMR graphs from text-based cache")
    return cache_dict


def save_smatch_cache_by_pairs(pairs_to_score_dict: dict, cache_path: str) -> None:
    """Save SMATCH scores indexed by unique text pairs.
    
    Parameters
    ----------
    pairs_to_score_dict : dict
        Dictionary with (text_a, text_b) tuple keys and F1 score values.
    cache_path : str
        Path to save the cache CSV file.
    """
    rows = []
    for (text_a, text_b), score in pairs_to_score_dict.items():
        rows.append({
            'TEXT_A': text_a,
            'TEXT_B': text_b,
            'SMATCH_F1': score
        })
    
    df = pd.DataFrame(rows)
    utils.ensure_output_dir(os.path.dirname(cache_path))
    df.to_csv(cache_path, index=False)


def load_smatch_cache_by_pairs(cache_path: str) -> dict:
    """Load SMATCH scores indexed by unique text pairs.
    
    Parameters
    ----------
    cache_path : str
        Path to the cache CSV file.
    
    Returns
    -------
    dict
        Dictionary with (text_a, text_b) tuple keys and F1 score values.
    """
    if not os.path.exists(cache_path):
        return {}
    
    df = pd.read_csv(cache_path)
    cache_dict = {}
    
    for _, row in df.iterrows():
        text_a = str(row['TEXT_A']) if pd.notna(row['TEXT_A']) else ""
        text_b = str(row['TEXT_B']) if pd.notna(row['TEXT_B']) else ""
        score = float(row['SMATCH_F1']) if pd.notna(row['SMATCH_F1']) else 0.0
        
        # Store with sorted tuple to handle (a,b) vs (b,a)
        pair_key = tuple(sorted([text_a, text_b]))
        cache_dict[pair_key] = score
    
    print(f"✓ Loaded {len(cache_dict)} unique text pair SMATCH scores from cache")
    return cache_dict


def migrate_old_cache_to_text_based(old_cache_path: str, new_cache_path: str, df: pd.DataFrame) -> dict:
    """Migrate old agent-based AMR cache to new text-based cache.
    
    Parameters
    ----------
    old_cache_path : str
        Path to old cache file (by agent, video, qnum).
    new_cache_path : str
        Path to new text-based cache file.
    df : pd.DataFrame
        Main dataframe with AGENT, VIDEO, QUESTION_NUM, ANSWER columns.
    
    Returns
    -------
    dict
        Text-based AMR cache dictionary.
    """
    print("\n--- Migrating old AMR cache to text-based format ---")
    
    # Load old cache
    old_cache = load_amr_cache(old_cache_path)
    
    if not old_cache:
        print("  No old cache found to migrate")
        return {}
    
    print(f"  Found {len(old_cache)} entries in old cache")
    
    # Build text-based cache
    text_cache = {}
    migrated_count = 0
    duplicate_count = 0
    
    for (agent, video, qnum), amr_graph in old_cache.items():
        # Find corresponding answer text in dataframe
        matches = df[(df['AGENT'] == agent) & 
                     (df['VIDEO'] == video) & 
                     (df['QUESTION_NUM'] == qnum)]
        
        if len(matches) > 0:
            answer_text = str(matches.iloc[0]['ANSWER'])
            normalized_text = normalize_answer_text(answer_text)
            
            if normalized_text not in text_cache:
                text_cache[normalized_text] = amr_graph
                migrated_count += 1
            else:
                duplicate_count += 1
    
    print(f"  Migrated {migrated_count} unique texts")
    print(f"  Found {duplicate_count} duplicate texts (saved {duplicate_count} redundant parses)")
    print(f"  Reduction: {duplicate_count / len(old_cache) * 100:.1f}% fewer entries")
    
    # Save new cache
    save_amr_cache_by_text(text_cache, new_cache_path)
    print(f"  ✓ Saved migrated cache to {os.path.basename(new_cache_path)}")
    
    return text_cache


def load_amr_parser():
    """Load AMR parser model."""
    print("Loading AMR parser model...")
    try:
        stog = amrlib.load_stog_model(model_dir="smatch_temp/model_parse_xfm_bart_base-v0_1_0")
        print("✓ AMR parser loaded")
        return stog
    except Exception as e:
        print(f"✗ Error loading AMR parser: {e}")
        print("Please ensure amrlib model is installed. Check amrlib installation instructions.")
        raise


def parse_to_amr(stog, text: str) -> str:
    """Parse text to AMR graph."""
    try:
        graphs = stog.parse_sents([text])
        return graphs[0] if graphs else ""
    except Exception as e:
        print(f"Warning: Failed to parse text: {text[:50]}... Error: {e}")
        return ""

def clean_amr_text(entry):
    text = re.sub(r'#.*\n', '', entry)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_smatch_score(amr1: str, amr2: str) -> float:
    """Compute SMATCH score between two AMR graphs."""
    if not amr1 or not amr2:
        return 0.0
    try:
        amr1 = clean_amr_text(amr1)
        amr2 = clean_amr_text(amr2)
        # Parse AMR strings
        from amrlib.evaluate.smatch_enhanced import match_pair,smatch
        score = match_pair((amr1, amr2))
        _,_,f1 = smatch.compute_f(*score)
        return f1
    except Exception as e:
        print(f"Warning: SMATCH computation failed: {e}")
        return 0.0


def compute_smatch_scores(skip_amr_parsing: bool = False) -> pd.DataFrame:
    """Compute pairwise SMATCH scores for all agent pairs with optimized AMR parsing.
    
    This function uses text-based deduplication to minimize redundant operations:
    1. Parse only unique answer texts to AMR graphs (GPU, batched, with checkpointing)
    2. Compute SMATCH only for unique text pairs (CPU, threaded, with checkpointing)
    3. Expand results to all agent pairs based on their answer texts
    
    Parameters
    ----------
    skip_amr_parsing : bool, optional
        If True, skip AMR parsing and only load from existing cache.
        Useful when you only need to recompute SMATCH scores.
        Default is False.
    
    Returns
    -------
    pd.DataFrame
        Pairwise scores with VIDEO, QUESTION_NUM, AGENT_I, AGENT_J, score columns.
    """
    print("\n=== Computing SMATCH Scores (Optimized with Text Deduplication) ===")
    
    if skip_amr_parsing:
        print("  [MODE] Skipping AMR parsing - loading from cache only")
    
    # Check if text deduplication is enabled
    #use_text_dedup = config.SMATCH_CONFIG.get("use_text_deduplication", True)
    #
    #if not use_text_dedup:
    #    print("  Text deduplication disabled, using legacy method...")
    #    return compute_smatch_scores_legacy()
    
    # Load AMR parser only if needed
    if not skip_amr_parsing:
        stog = load_amr_parser()
    else:
        stog = None
    
    # Load answers
    df = utils.load_answers()
    agents = utils.get_agent_groups()["all"]
    
    # Normalize all answer texts
    if config.SMATCH_CONFIG.get("normalize_text", True):
        print("\nNormalizing answer texts...")
        df['ANSWER_NORMALIZED'] = df['ANSWER'].apply(normalize_answer_text)
    else:
        df['ANSWER_NORMALIZED'] = df['ANSWER']
    
    # Statistics
    total_answers = len(df)
    unique_texts = df['ANSWER_NORMALIZED'].nunique()
    duplication_rate = (1 - unique_texts / total_answers) * 100
    
    print(f"\nDataset statistics:")
    print(f"  Total answers: {total_answers:,}")
    print(f"  Unique texts: {unique_texts:,}")
    print(f"  Duplication rate: {duplication_rate:.2f}%")
    print(f"  AMR parsing operations saved: {total_answers - unique_texts:,} ({duplication_rate:.1f}%)")
    
    # ========== PHASE 1: Parse unique texts to AMR graphs ==========
    print("\n--- Phase 1: Parsing unique answer texts to AMR graphs ---")
    
    # Load existing AMR cache by text
    amr_cache_by_text = load_amr_cache_by_text(config.SMATCH_SCORES["amr_cache_by_text"])
    initial_cache_size = len(amr_cache_by_text)
    
    # Identify missing unique texts
    all_unique_texts = df['ANSWER_NORMALIZED'].unique().tolist()
    missing_texts = [text for text in all_unique_texts if text not in amr_cache_by_text]
    
    print(f"  Found {initial_cache_size}/{unique_texts} unique texts in cache")
    print(missing_texts[:5])  # Show sample of missing texts
    print(f"  Need to parse {len(missing_texts)} new unique texts")
    
    # Parse missing unique texts in batches
    if len(missing_texts) > 0:
        if skip_amr_parsing:
            print(f"  ⚠ SKIP MODE: {len(missing_texts)} missing texts will not be parsed")
            print(f"    Only cached texts will be used ({initial_cache_size} texts)")
            for text in missing_texts:
                amr_cache_by_text[text] = ""
            print(f"✓ Phase 1 complete: {len(amr_cache_by_text)} unique texts in cache (skip mode)")
        else:
            BATCH_SIZE = config.SMATCH_CONFIG["batch_size_amr"]
            CHECKPOINT_FREQ = config.SMATCH_CONFIG["checkpoint_amr"]
            pbar = tqdm.tqdm(range(0, len(missing_texts), BATCH_SIZE), desc="Parsing AMR", unit="batch")
            for i in pbar:
                batch_texts = missing_texts[i:i+BATCH_SIZE]
                
                # Parse batch (single GPU call)
                try:
                    batch_graphs = stog.parse_sents(batch_texts)
                    
                    # Update cache with results
                    for text, graph in zip(batch_texts, batch_graphs):
                        amr_cache_by_text[text] = graph if graph else ""
                    
                except Exception as e:
                    print(f"  Warning: Batch parsing failed: {e}")
                    # Add empty graphs for failed parses
                    for text in batch_texts:
                        amr_cache_by_text[text] = ""
                
                # Progress update
                #parsed_so_far = min(i + BATCH_SIZE, len(missing_texts))
                #print(f"  Parsed {initial_cache_size + parsed_so_far}/{unique_texts} unique texts ({parsed_so_far}/{len(missing_texts)} new)")
                
                # Checkpoint: save cache periodically
                if (i + BATCH_SIZE) % CHECKPOINT_FREQ == 0 or (i + BATCH_SIZE) >= len(missing_texts):
                    save_amr_cache_by_text(amr_cache_by_text, config.SMATCH_SCORES["amr_cache_by_text"])
                    print(f"  ✓ Checkpoint: AMR cache saved ({len(amr_cache_by_text)} unique texts)")
            
            print(f"✓ Phase 1 complete: {len(amr_cache_by_text)} total unique AMR graphs in cache")
    else:
        print(f"✓ Phase 1 complete: All unique texts already cached")
    
    # ========== PHASE 2: Compute SMATCH scores for unique text pairs ==========
    print("\n--- Phase 2: Computing SMATCH scores for unique text pairs ---")
    
    # Create mapping: (agent, video, qnum) -> normalized_text
    answer_text_map = {}
    for _, row in df.iterrows():
        key = (row['AGENT'], row['VIDEO'], row['QUESTION_NUM'])
        answer_text_map[key] = row['ANSWER_NORMALIZED']
    
    # Load existing SMATCH scores by text pairs
    smatch_cache_by_pairs = load_smatch_cache_by_pairs(config.SMATCH_SCORES["smatch_cache_by_pairs"])
    initial_smatch_cache_size = len(smatch_cache_by_pairs)
    
    # Identify all unique text pairs needed for agent comparisons
    needed_text_pairs = set()
    agent_pair_to_text_pair = {}
    total_agent_pairs = 0
    
    for (video, question_num), group in df.groupby(["VIDEO", "QUESTION_NUM"]):
        available_agents = group["AGENT"].unique()
        
        for agent_i, agent_j in combinations(agents, 2):
            if agent_i not in available_agents or agent_j not in available_agents:
                continue
            
            total_agent_pairs += 1
            
            key_i = (agent_i, video, question_num)
            key_j = (agent_j, video, question_num)
            
            text_i = answer_text_map.get(key_i, "")
            text_j = answer_text_map.get(key_j, "")
            
            # Create sorted pair key to handle (a,b) vs (b,a)
            text_pair = tuple(sorted([text_i, text_j]))
            needed_text_pairs.add(text_pair)
            
            # Store mapping for later expansion
            agent_pair_key = (video, question_num, agent_i, agent_j)
            agent_pair_to_text_pair[agent_pair_key] = (text_i, text_j)
    
    unique_text_pairs_needed = len(needed_text_pairs)
    missing_text_pairs = [pair for pair in needed_text_pairs if pair not in smatch_cache_by_pairs]
    
    smatch_reduction = (1 - unique_text_pairs_needed / total_agent_pairs) * 100 if total_agent_pairs > 0 else 0
    
    print(f"  Total agent pairs to compare: {total_agent_pairs:,}")
    print(f"  Unique text pairs needed: {unique_text_pairs_needed:,}")
    print(f"  SMATCH comparisons saved: {total_agent_pairs - unique_text_pairs_needed:,} ({smatch_reduction:.1f}%)")
    print(f"  Found {initial_smatch_cache_size} text pairs in cache")
    print(f"  Need to compute {len(missing_text_pairs)} new text pairs")
    
    # Compute missing text pairs with threading
    if len(missing_text_pairs) > 0:
        MAX_WORKERS = config.SMATCH_CONFIG["max_workers"]
        CHECKPOINT_FREQ = config.SMATCH_CONFIG["checkpoint_scores"]
        
        computed_count = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all comparison tasks
            futures = {}
            for text_a, text_b in missing_text_pairs:
                # Handle self-comparison
                if text_a == text_b:
                    smatch_cache_by_pairs[(text_a, text_b)] = 1.0
                    continue
                
                amr_a = amr_cache_by_text.get(text_a, "")
                amr_b = amr_cache_by_text.get(text_b, "")
                
                future = executor.submit(
                    compute_smatch_score,
                    amr_a,
                    amr_b
                )
                futures[future] = (text_a, text_b)
            
            # Collect results as they complete
            for future in as_completed(futures):
                text_pair = futures[future]
                try:
                    score = future.result()
                    smatch_cache_by_pairs[text_pair] = score
                except Exception as e:
                    print(f"  Warning: SMATCH comparison failed: {e}")
                    smatch_cache_by_pairs[text_pair] = 0.0
                
                computed_count += 1
                
                # Progress update
                if computed_count % 100 == 0:
                    print(f"  Computed {initial_smatch_cache_size + computed_count}/{unique_text_pairs_needed} text pairs ({computed_count}/{len(missing_text_pairs)} new)")
                
                # Checkpoint: save cache periodically
                if computed_count % CHECKPOINT_FREQ == 0:
                    save_smatch_cache_by_pairs(smatch_cache_by_pairs, config.SMATCH_SCORES["smatch_cache_by_pairs"])
                    print(f"  ✓ Checkpoint: {len(smatch_cache_by_pairs)} text pair scores saved")
        
        # Final save
        save_smatch_cache_by_pairs(smatch_cache_by_pairs, config.SMATCH_SCORES["smatch_cache_by_pairs"])
        print(f"✓ Phase 2 complete: {len(smatch_cache_by_pairs)} total text pair scores in cache")
    else:
        print(f"✓ Phase 2 complete: All text pairs already computed")
    
    # ========== PHASE 3: Expand to all agent pairs ==========
    print("\n--- Phase 3: Expanding results to all agent pairs ---")
    
    results = []
    for (video, question_num, agent_i, agent_j), (text_i, text_j) in agent_pair_to_text_pair.items():
        # Get score from cache (using sorted pair key)
        text_pair = tuple(sorted([text_i, text_j]))
        score = smatch_cache_by_pairs.get(text_pair, 0.0)
        
        results.append({
            "VIDEO": video,
            "QUESTION_NUM": question_num,
            "AGENT_I": agent_i,
            "AGENT_J": agent_j,
            "score": score
        })
    
    pairwise_df = pd.DataFrame(results)
    
    print(f"  Expanded to {len(pairwise_df):,} agent pair comparisons")
    
    # Save pairwise scores
    utils.ensure_output_dir(config.OUTPUT_SMATCH_DIR)
    pairwise_df.to_csv(config.SMATCH_SCORES["pairwise"], index=False)
    print(f"✓ Saved: {config.SMATCH_SCORES['pairwise']}")
    
    # Show efficiency summary
    print("\n✓ OPTIMIZATION SUMMARY:")
    print(f"  AMR parsing: {total_answers:,} answers → {unique_texts:,} unique texts")
    print(f"    Saved: {total_answers - unique_texts:,} parsing operations ({duplication_rate:.1f}%)")
    print(f"  SMATCH comparison: {total_agent_pairs:,} agent pairs → {unique_text_pairs_needed:,} text pairs")
    print(f"    Saved: {total_agent_pairs - unique_text_pairs_needed:,} comparisons ({smatch_reduction:.1f}%)")
    
    return pairwise_df


def aggregate_smatch_scores(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SMATCH scores by blocks."""
    print("\n=== Aggregating SMATCH Scores by Blocks ===")
    
    aggregated_df = utils.aggregate_scores_by_block(pairwise_df, score_column="score")
    
    # Save aggregated scores
    aggregated_df.to_csv(config.SMATCH_SCORES["aggregated"], index=False)
    print(f"✓ Saved: {config.SMATCH_SCORES['aggregated']}")
    
    return aggregated_df


def generate_smatch_heatmaps(aggregated_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> None:
    """Generate similarity and agreement heatmaps for SMATCH scores.
    
    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Aggregated scores with MEAN_SCORE column.
    pairwise_df : pd.DataFrame
        Pairwise scores for computing agreement percentages.
    """
    print("\n=== Generating SMATCH Heatmaps ===")
    
    utils.ensure_output_dir(config.OUTPUT_SMATCH_PLOTS)
    
    # Generate color map
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cmap", ["#ffffff", "#6be425"])
    
    region_labels = {"both": "All", "lima": "Lima", "nyc": "NYC"}
    total_plots = 0

    for block in range(1, config.NUM_BLOCKS + 1):
        for region in ["both", "lima", "nyc"]:
            # Similarity heatmap
            similarity_matrix = create_similarity_matrix(
                aggregated_df,
                block,
                score_column="score",
                video_region=region,
                pairwise_df=pairwise_df,
            )

            if region == "both":
                similarity_path = os.path.join(config.OUTPUT_SMATCH_PLOTS, f"smatch_similarity_block{block}.png")
            else:
                similarity_path = os.path.join(config.OUTPUT_SMATCH_PLOTS, f"smatch_similarity_block{block}_{region}.png")

            region_title = "" if region == "both" else f" ({region_labels[region]})"
            plot_similarity_heatmap(
                similarity_matrix,
                title=f"SMATCH F1 Score - Block {block}{region_title}",
                color_label="Similarity Score",
                output_path=similarity_path,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                use_group_colors=True
            )
            if similarity_matrix is not None and not similarity_matrix.empty:
                total_plots += 1

            # Agreement heatmap
            agreement_matrix = create_agreement_matrix(
                pairwise_df,
                block,
                score_column="score",
                threshold=0.5,
                video_region=region,
            )

            if region == "both":
                agreement_path = os.path.join(config.OUTPUT_SMATCH_PLOTS, f"smatch_agreement_block{block}.png")
            else:
                agreement_path = os.path.join(config.OUTPUT_SMATCH_PLOTS, f"smatch_agreement_block{block}_{region}.png")

            plot_similarity_heatmap(
                agreement_matrix,
                title=f"SMATCH F1 Score Agrement - Block {block}{region_title}",
                color_label="Agreement (%)",
                output_path=agreement_path,
                cmap=cmap,
                vmin=0,
                vmax=100,
                fmt="%d",
                use_group_colors=True
            )
            if agreement_matrix is not None and not agreement_matrix.empty:
                total_plots += 1
    
    print(f"✓ Generated {total_plots} SMATCH heatmap plots")


def main():
    """Main execution: compute scores if needed, then generate heatmaps."""
    print("=" * 60)
    print("SMATCH SCORE HEATMAP ANALYSIS (AMR-based)")
    print("=" * 60)
    
    use_text_dedup = config.SMATCH_CONFIG.get("use_text_deduplication", True)
    
    # Check if caches and scores exist
    amr_cache_exists = os.path.exists(config.SMATCH_SCORES["amr_cache"])
    amr_cache_by_text_exists = os.path.exists(config.SMATCH_SCORES["amr_cache_by_text"])
    smatch_cache_by_pairs_exists = os.path.exists(config.SMATCH_SCORES["smatch_cache_by_pairs"])
    pairwise_exists = os.path.exists(config.SMATCH_SCORES["pairwise"])
    aggregated_exists = os.path.exists(config.SMATCH_SCORES["aggregated"])
    
    # Handle cache migration if needed
    if use_text_dedup and amr_cache_exists and not amr_cache_by_text_exists:
        print("\n" + "=" * 60)
        print("CACHE MIGRATION DETECTED")
        print("=" * 60)
        print("\nFound old agent-based AMR cache.")
        print("Migrating to optimized text-based cache format...")
        print("This will deduplicate entries and save disk space.")
        
        # Load dataset for migration
        df = utils.load_answers()
        
        # Migrate
        migrate_old_cache_to_text_based(
            config.SMATCH_SCORES["amr_cache"],
            config.SMATCH_SCORES["amr_cache_by_text"],
            df
        )
        
        amr_cache_by_text_exists = True
        print("\n✓ Migration complete!")
    
    # Show cache statistics if available
    print("\n" + "-" * 60)
    print("CACHE STATUS")
    print("-" * 60)
    
    if use_text_dedup:
        if amr_cache_by_text_exists:
            amr_df = pd.read_csv(config.SMATCH_SCORES["amr_cache_by_text"])
            print(f"✓ Found AMR cache (text-based): {len(amr_df):,} unique texts")
        else:
            print("  No AMR cache found (will parse from scratch)")
        
        if smatch_cache_by_pairs_exists:
            smatch_df = pd.read_csv(config.SMATCH_SCORES["smatch_cache_by_pairs"])
            print(f"✓ Found SMATCH cache (text pairs): {len(smatch_df):,} unique pairs")
        else:
            print("  No SMATCH pair cache found (will compute from scratch)")
    else:
        if amr_cache_exists:
            amr_df = pd.read_csv(config.SMATCH_SCORES["amr_cache"])
            print(f"✓ Found AMR cache (legacy): {len(amr_df):,} entries")
        else:
            print("  No AMR cache found (will parse from scratch)")
    
    if pairwise_exists:
        pairwise_df = pd.read_csv(config.SMATCH_SCORES["pairwise"])
        print(f"✓ Found pairwise scores: {len(pairwise_df):,} agent pair comparisons")
    else:
        print("  No pairwise scores found")
    
    print("-" * 60)
    
    # Compute or load scores
    # Check for new agents even if pairwise exists
    new_agents = set()
    if pairwise_exists:
        new_agents = utils.detect_new_agents(config.SMATCH_SCORES["pairwise"])
    
    needs_recompute = not pairwise_exists or len(new_agents) > 0
    
    if needs_recompute:
        if new_agents:
            print(f"\n⚡ New agents detected: {sorted(new_agents)}")
            print("  Will recompute pairwise scores (AMR + SMATCH caches will be reused)")
        else:
            print("\n⚠ Pairwise SMATCH scores not found. Computing (this may take a while)...")
        
        if use_text_dedup:
            print("  Using OPTIMIZED method (text deduplication enabled)")
            if amr_cache_by_text_exists or smatch_cache_by_pairs_exists:
                print("  (Will leverage existing caches)")
        else:
            print("  Using LEGACY method (text deduplication disabled)")
            if amr_cache_exists:
                print("  (Will use existing AMR cache)")
        
        # Auto-skip AMR parsing if cache exists and we're just recomputing SMATCH
        skip_amr = amr_cache_by_text_exists and not new_agents
        if skip_amr:
            print("  ✓ Using cached AMR graphs (skipping parsing phase)")
        
        pairwise_df = compute_smatch_scores(skip_amr_parsing=skip_amr)
        aggregated_df = aggregate_smatch_scores(pairwise_df)
    elif not aggregated_exists:
        print("\n⚠ Aggregated SMATCH scores not found. Aggregating...")
        pairwise_df = pd.read_csv(config.SMATCH_SCORES["pairwise"])
        aggregated_df = aggregate_smatch_scores(pairwise_df)
    else:
        print("\n✓ SMATCH scores already computed. Loading...")
        pairwise_df = pd.read_csv(config.SMATCH_SCORES["pairwise"])
        aggregated_df = pd.read_csv(config.SMATCH_SCORES["aggregated"])
    
    # Generate heatmaps
    generate_smatch_heatmaps(aggregated_df, pairwise_df)
    
    print("\n" + "=" * 60)
    print("✓ SMATCH analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
