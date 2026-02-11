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


def compute_stsb_scores() -> pd.DataFrame:
    """Compute pairwise STSB-RoBERTa scores for all agent pairs."""
    print("\n=== Computing STSB-RoBERTa Scores ===")
    
    # Load model
    model = load_stsb_model()
    
    # Load answers
    df = utils.load_answers()
    agents = utils.get_agent_groups()["all"]
    
    results = []
    total_comparisons = 0
    
    # Prepare batches for efficient computation
    batch_pairs = []
    batch_metadata = []
    
    for (video, question_num), group in df.groupby(["VIDEO", "QUESTION_NUM"]):
        video_data = {row["AGENT"]: row["ANSWER"] for _, row in group.iterrows()}
        
        # Collect pairs for this video/question
        for agent_i, agent_j in combinations(agents, 2):
            if agent_i not in video_data or agent_j not in video_data:
                continue
            
            answer_i = str(video_data[agent_i])
            answer_j = str(video_data[agent_j])
            
            batch_pairs.append([answer_i, answer_j])
            batch_metadata.append({
                "VIDEO": video,
                "QUESTION_NUM": question_num,
                "AGENT_I": agent_i,
                "AGENT_J": agent_j
            })
    
    print(f"  Prepared {len(batch_pairs)} pairs for scoring...")
    
    # Compute scores in batches
    batch_size = 32
    all_scores = []
    
    for i in range(0, len(batch_pairs), batch_size):
        batch = batch_pairs[i:i+batch_size]
        scores = model.predict(batch)
        all_scores.extend(scores)
        
        if (i + batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(batch_pairs))}/{len(batch_pairs)} pairs...")
    
    print(f"✓ Computed {len(all_scores)} STSB-RoBERTa score comparisons")
    
    # Combine with metadata
    for metadata, score in zip(batch_metadata, all_scores):
        metadata["BERT_SCORE"] = float(score) # Normalize to 0-1 range (STSB is 0-5)
        results.append(metadata)
    
    pairwise_df = pd.DataFrame(results)
    
    # Save pairwise scores
    utils.ensure_output_dir(config.OUTPUT_STSB_DIR)
    pairwise_df.to_csv(config.STSB_SCORES["pairwise"], index=False)
    print(f"✓ Saved: {config.STSB_SCORES['pairwise']}")
    
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
    """Main execution: compute scores if needed, then generate heatmaps."""
    print("=" * 60)
    print("STSB-RoBERTa SCORE HEATMAP ANALYSIS")
    print("=" * 60)
    
    # Check if scores exist
    pairwise_exists = os.path.exists(config.STSB_SCORES["pairwise"])
    aggregated_exists = os.path.exists(config.STSB_SCORES["aggregated"])
    
    if not pairwise_exists:
        print("\n⚠ Pairwise STSB-RoBERTa scores not found. Computing...")
        pairwise_df = compute_stsb_scores()
        aggregated_df = aggregate_stsb_scores(pairwise_df)
    elif not aggregated_exists:
        print("\n⚠ Aggregated STSB-RoBERTa scores not found. Aggregating...")
        pairwise_df, _ = utils.load_metric_scores(config.STSB_SCORES["pairwise"])
        aggregated_df = aggregate_stsb_scores(pairwise_df)
    else:
        print("\n✓ STSB-RoBERTa scores already computed. Loading...")
        _, aggregated_df = utils.load_metric_scores(
            config.STSB_SCORES["pairwise"],
            config.STSB_SCORES["aggregated"]
        )
    
    # Generate heatmaps
    generate_stsb_heatmaps(aggregated_df,pairwise_df)
    
    print("\n" + "=" * 60)
    print("✓ STSB-RoBERTa analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
