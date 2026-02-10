"""SMATCH Score heatmap analysis with AMR parsing and auto-computation."""
import os
import pandas as pd
from itertools import combinations
import amrlib
from src import config, utils
from src.heatmap_common import create_similarity_matrix, plot_similarity_heatmap, plot_agreement_heatmap


def load_amr_parser():
    """Load AMR parser model."""
    print("Loading AMR parser model...")
    try:
        stog = amrlib.load_stog_model()
        print("✓ AMR parser loaded")
        return stog
    except Exception as e:
        print(f"✗ Error loading AMR parser: {e}")
        print("Please ensure amrlib model is installed:")
        print("  python -m amrlib.download model_parse_xfm_bart_large")
        raise


def parse_to_amr(stog, text: str) -> str:
    """Parse text to AMR graph."""
    try:
        graphs = stog.parse_sents([text])
        return graphs[0] if graphs else ""
    except Exception as e:
        print(f"Warning: Failed to parse text: {text[:50]}... Error: {e}")
        return ""


def compute_smatch_score(amr1: str, amr2: str) -> float:
    """Compute SMATCH score between two AMR graphs."""
    if not amr1 or not amr2:
        return 0.0
    
    try:
        import smatch
        # Parse AMR strings
        from smatch import score_amr_pairs
        precision, recall, f1 = score_amr_pairs(
            [(amr1, amr2)],
            verbose=False
        )
        return f1
    except Exception as e:
        print(f"Warning: SMATCH computation failed: {e}")
        return 0.0


def compute_smatch_scores() -> pd.DataFrame:
    """Compute pairwise SMATCH scores for all agent pairs with AMR parsing."""
    print("\n=== Computing SMATCH Scores (with AMR Parsing) ===")
    
    # Load AMR parser
    stog = load_amr_parser()
    
    # Load answers
    df = utils.load_answers()
    agents = utils.get_agent_groups()["all"]
    
    # Cache AMR parses to avoid recomputation
    print("\nStep 1: Parsing all answers to AMR graphs...")
    amr_cache = {}
    total_parses = 0
    
    for idx, row in df.iterrows():
        key = (row["AGENT"], row["VIDEO"], row["QUESTION_NUM"])
        amr_cache[key] = parse_to_amr(stog, str(row["ANSWER"]))
        total_parses += 1
        if total_parses % 100 == 0:
            print(f"  Parsed {total_parses}/{len(df)} answers...")
    
    print(f"✓ Parsed {total_parses} answers to AMR graphs")
    
    # Compute pairwise SMATCH scores
    print("\nStep 2: Computing SMATCH scores for agent pairs...")
    results = []
    total_comparisons = 0
    
    for (video, question_num), group in df.groupby(["VIDEO", "QUESTION_NUM"]):
        available_agents = group["AGENT"].unique()
        
        # Compute pairwise scores
        for agent_i, agent_j in combinations(agents, 2):
            if agent_i not in available_agents or agent_j not in available_agents:
                continue
            
            key_i = (agent_i, video, question_num)
            key_j = (agent_j, video, question_num)
            
            amr_i = amr_cache.get(key_i, "")
            amr_j = amr_cache.get(key_j, "")
            
            # Compute SMATCH F1 score
            smatch_f1 = compute_smatch_score(amr_i, amr_j)
            
            results.append({
                "VIDEO": video,
                "QUESTION_NUM": question_num,
                "AGENT_I": agent_i,
                "AGENT_J": agent_j,
                "score": smatch_f1
            })
            
            total_comparisons += 1
            if total_comparisons % 100 == 0:
                print(f"  Processed {total_comparisons} comparisons...")
    
    print(f"✓ Computed {total_comparisons} SMATCH score comparisons")
    
    pairwise_df = pd.DataFrame(results)
    
    # Save pairwise scores
    utils.ensure_output_dir(config.OUTPUT_SMATCH_DIR)
    pairwise_df.to_csv(config.SMATCH_SCORES["pairwise"], index=False)
    print(f"✓ Saved: {config.SMATCH_SCORES['pairwise']}")
    
    return pairwise_df


def aggregate_smatch_scores(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SMATCH scores by blocks."""
    print("\n=== Aggregating SMATCH Scores by Blocks ===")
    
    aggregated_df = utils.aggregate_scores_by_block(pairwise_df, score_column="score")
    
    # Save aggregated scores
    aggregated_df.to_csv(config.SMATCH_SCORES["aggregated"], index=False)
    print(f"✓ Saved: {config.SMATCH_SCORES['aggregated']}")
    
    return aggregated_df


def generate_smatch_heatmaps(aggregated_df: pd.DataFrame) -> None:
    """Generate similarity and agreement heatmaps for SMATCH scores."""
    print("\n=== Generating SMATCH Heatmaps ===")
    
    utils.ensure_output_dir(config.OUTPUT_SMATCH_PLOTS)
    
    for block in range(1, config.NUM_BLOCKS + 1):
        matrix = create_similarity_matrix(aggregated_df, block, score_column="score")
        
        # Similarity heatmap
        similarity_path = os.path.join(
            config.OUTPUT_SMATCH_PLOTS,
            f"smatch_similarity_block{block}.png"
        )
        
        #generate color map linearly
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list([
            (0.0, "white"),    # 0.0 -> red
            (1.0, "green")]   # 1.0 -> green
        )
        plot_similarity_heatmap(
            matrix,
            title=f"SMATCH Similarity - Block {block}",
            output_path=similarity_path,
            cmap=cmap,  # Different colormap for SMATCH
            vmin=0.0,
            vmax=1.0
        )
        
        # Agreement heatmap
        agreement_path = os.path.join(
            config.OUTPUT_SMATCH_PLOTS,
            f"smatch_agreement_block{block}.png"
        )
        plot_agreement_heatmap(
            matrix,
            title=f"SMATCH Agreement - Block {block}",
            output_path=agreement_path,
            thresholds={"high": 0.7, "medium": 0.4}  # Custom thresholds for SMATCH
        )
    
    print(f"✓ Generated {config.NUM_BLOCKS * 2} SMATCH heatmap plots")


def main():
    """Main execution: compute scores if needed, then generate heatmaps."""
    print("=" * 60)
    print("SMATCH SCORE HEATMAP ANALYSIS (AMR-based)")
    print("IF RUNNING FOR THE FIRST TIME, MAKE SURE TO ")
    print("=" * 60)
    
    # Check if scores exist
    pairwise_exists = os.path.exists(config.SMATCH_SCORES["pairwise"])
    aggregated_exists = os.path.exists(config.SMATCH_SCORES["aggregated"])
    
    if not pairwise_exists:
        print("\n⚠ Pairwise SMATCH scores not found. Computing (this may take a while)...")
        pairwise_df = compute_smatch_scores()
        aggregated_df = aggregate_smatch_scores(pairwise_df)
    elif not aggregated_exists:
        print("\n⚠ Aggregated SMATCH scores not found. Aggregating...")
        pairwise_df, _ = utils.load_metric_scores(config.SMATCH_SCORES["pairwise"])
        aggregated_df = aggregate_smatch_scores(pairwise_df)
    else:
        print("\n✓ SMATCH scores already computed. Loading...")
        _, aggregated_df = utils.load_metric_scores(
            config.SMATCH_SCORES["pairwise"],
            config.SMATCH_SCORES["aggregated"]
        )
    
    # Generate heatmaps
    generate_smatch_heatmaps(aggregated_df)
    
    print("\n" + "=" * 60)
    print("✓ SMATCH analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
