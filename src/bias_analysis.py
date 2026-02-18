"""Bias analysis comparing VLM ratings against human ratings by region.

Focus: Block 2 (Q6-Q10) which contains numerical ratings.
Method: Unit consensus - compare each VLM against each human individually,
        get sign (+1, -1, 0), then average signs.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from src import config, utils


def extract_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numerical ratings from answer texts (Block 2: Q6-Q10 only)."""
    print("\n=== Extracting Ratings from Answers ===")
    
    # Filter Block 2 only (questions 6-10)
    block2_questions = [6, 7, 8, 9, 10]
    df_block2 = df[df["QUESTION_NUM"].isin(block2_questions)].copy()
    print(f"  Filtering Block 2 (Q6-Q10): {len(df_block2)} rows")
    
    def extract_number(text):
        """Extract first number found in text (handles ratings, scores, etc.)."""
        text = str(text)
        # Look for patterns like: "5", "3.5", "rating: 4", "score of 7", etc.
        patterns = [
            r'rating[:\s]+(\d+\.?\d*)',
            r'score[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:out of|/)\s*\d+',
            r'^(\d+\.?\d*)',
            r'(\d+\.?\d*)'
            r'(\d+\.)' #
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return np.nan
    
    df_block2["rating"] = df_block2["ANSWER"].apply(extract_number)
    
    # Extract canonical video region columns
    df_block2 = utils.add_video_region_columns(
        df_block2,
        video_col="VIDEO",
        id_col="video_id",
        region_col="video_region"
    )
    
    # Filter out rows without ratings
    original_len = len(df_block2)
    df_block2 = df_block2.dropna(subset=["rating"])
    extracted_len = len(df_block2)
    
    print(f"✓ Extracted {extracted_len}/{original_len} ratings ({extracted_len/original_len:.1%})")
    print(f"  Video regions: Lima={len(df_block2[df_block2['video_region']=='lima'])}, NYC={len(df_block2[df_block2['video_region']=='nyc'])}")
    
    return df_block2


def compute_unit_consensus_bias(df: pd.DataFrame, human_region: str = "lima", video_region: str = "lima") -> pd.DataFrame:
    """
    Compute bias using unit consensus method:
    1. Compare each VLM against EACH human individually
    2. Assign sign: +1 (VLM > human), -1 (VLM < human), 0 (equal)
    3. Average signs across all humans in the region
    
    Args:
        human_region: "lima" or "nyc" (which human annotators)
        video_region: "lima" or "nyc" or "both" (which video set: 1-100 or 101-200)
    
    Returns:
        DataFrame with columns: VIDEO, QUESTION_NUM, AGENT (VLM), avg_bias_sign
    """
    print(f"\n=== Computing Unit Consensus Bias (Humans: {human_region.upper()}, Videos: {video_region.upper()}) ===")
    
    # Filter humans by region
    if human_region == "lima":
        human_agents = config.LIMA_AGENTS
    elif human_region == "nyc":
        human_agents = config.NYC_AGENTS
    else:
        raise ValueError("human_region must be 'lima' or 'nyc'")
    
    # Filter videos by region
    #video region can be both
    if video_region not in ["lima", "nyc", "both"]:
        raise ValueError("video_region must be 'lima', 'nyc', or 'both'")

    if video_region != "both":
        df = df[df["video_region"] == video_region].copy()
    
    if len(df) == 0:
        print(f"  ⚠ No videos found for region: {video_region}")
        return pd.DataFrame()
    
    vlm_agents = config.VLM_AGENTS
    
    # Filter dataframe
    human_df = df[df["AGENT"].isin(human_agents)].copy()
    vlm_df = df[df["AGENT"].isin(vlm_agents)].copy()
    
    print(f"  Human agents: {len(human_agents)}")
    print(f"  VLM agents: {len(vlm_agents)}")
    
    results = []
    
    # For each VLM, video, question combination
    for (vlm, video, qnum), vlm_row in vlm_df.groupby(["AGENT", "VIDEO", "QUESTION_NUM"]):
        vlm_rating = vlm_row["rating"].iloc[0]
        
        # Get all human ratings for this (video, question)
        human_ratings = human_df[
            (human_df["VIDEO"] == video) & 
            (human_df["QUESTION_NUM"] == qnum)
        ]["rating"].values
        
        if len(human_ratings) == 0:
            continue
        
        # Compute sign for each human comparison
        signs = []
        for human_rating in human_ratings:
            if vlm_rating > human_rating:
                signs.append(+1)
            elif vlm_rating < human_rating:
                signs.append(-1)
            else:
                signs.append(0)
        
        # Average the signs
        avg_sign = np.mean(signs)
        
        results.append({
            "VIDEO": video,
            "QUESTION_NUM": qnum,
            "AGENT": vlm,
            "avg_bias_sign": avg_sign,
            "num_comparisons": len(signs)
        })
    
    result_df = pd.DataFrame(results)
    print(f"✓ Computed {len(result_df)} bias values (unit consensus method)")
    
    return result_df

def get_region_colormap(region: str):
    # grayscale for everyone
    #use config plot colors for video regions
    return LinearSegmentedColormap.from_list(
        f"Colormap",
        [config.PLOT_COLORS["BASE"], config.PLOT_COLORS["OTHER"]],
    )
    
def plot_bias_heatmap_per_question(bias_df: pd.DataFrame, human_region: str, video_region: str, question_num: int, output_dir: str) -> None:
    """
    Plot single heatmap for one question: VLMs (Y) vs Videos (X).
    
    Args:
        bias_df: DataFrame with avg_bias_sign values
        human_region: "lima" or "nyc" (human annotators)
        video_region: "lima" or "nyc" (video set)
        question_num: Question number (6-10)
        output_dir: Directory to save plot
    """
    # Filter for this question
    q_df = bias_df[bias_df["QUESTION_NUM"] == question_num].copy()
    
    if len(q_df) == 0:
        print(f"  ⚠ No data for Q{question_num}")
        return
    
    # Pivot to matrix: VLMs × Videos
    heatmap_data = q_df.pivot_table(
        index="AGENT",
        columns="VIDEO",
        values="avg_bias_sign",
        aggfunc="mean"
    )
    cmap = get_region_colormap(video_region)
    #get n columns
    
    # Keep consistent VLM order across all plots (alphabetical)
    vlm_order = sorted(heatmap_data.index)
    heatmap_data = heatmap_data.loc[vlm_order]
    row_means = heatmap_data.mean(axis=1).to_frame(name="Average rating")
    
    # Adjust figure width based on number of videos
    num_videos = len(heatmap_data.columns)
    proportional_width = [1, num_videos]
    fig_width = max(10, min(20, num_videos * 0.8))
    fig, (ax_avg,ax_main) = plt.subplots(1,2, figsize=(fig_width, 8), sharey=True, gridspec_kw={'width_ratios': proportional_width})
    
    # move the labels to the right of the heatmap
    sns.heatmap(
        row_means,
        ax=ax_avg,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        yticklabels=True,
        cbar=False,
        linecolor="gray"
    )
    
    #put labels to the right of the heatmap
    ax_avg.tick_params(left=False, labelleft=False)
    ax_avg.tick_params(right=True, labelright=True)
    # set rotation of labels to 0
    ax_avg.set_yticklabels(ax_avg.get_yticklabels(), rotation=0, fontsize=10)
    


    sns.heatmap(
        heatmap_data,
        ax=ax_main,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        yticklabels=True,
        square=False,
        linewidths=0.5,
        cbar_kws={
            "label": "Avg Bias Sign\n(-1=underestimate, +1=overestimate)",
            "shrink": 1
        },
        linecolor="gray"
    )
    ax_main.set_ylabel("VLM Agent", fontsize=12)
        
    human_label = "Lima" if human_region == "lima" else "NYC"
    ax_main.set_title(
        f"Q{question_num} - VLM vs {human_label} Humans: Rating bias \n",
        fontsize=config.PLOT_CONFIG["title_fontsize"],
        pad=-30,
        fontweight='bold',
        
    )
    ax_main.set_xlabel("Video", fontsize=12)
    ax_main.set_ylabel("", fontsize=12)
    
    plt.xticks(rotation=45, ha="right", fontsize=8)
    
    #q: what doses tight layout does? explain it in 3 bullet points
    #r1: Automatically adjusts subplot parameters to give specified padding
    #r2: Prevents overlap of subplot elements (titles, labels, ticks)
    #r3: Ensures the entire figure fits within the specified figure size without clipping
    
    #q: why does it changes the title size
    #r: tight_layout can sometimes adjust the spacing in a way that affects the title size or position. If you want to maintain a specific title size, you can set it after calling tight_layout or adjust the layout parameters to prevent it from resizing the title.
    
    #plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"bias_heatmap_H{human_region}_V{video_region}_Q{question_num}.png")
    plt.tight_layout(pad=-2.5)
    utils.ensure_output_dir(output_dir)
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")

def plot_bias_heatmap_avg_per_question(bias_df: pd.DataFrame, human_region: str, video_region: str, output_dir: str) -> None:
    """
    Plot heatmap where each column is the average performance per question.

    Args:
        bias_df: DataFrame with avg_bias_sign values
        human_region: "lima" or "nyc" (human annotators)
        video_region: "lima" or "nyc" (video set)
        output_dir: Directory to save plot
    """
    print("\n=== Generating Heatmap: Average Performance per Question ===")

    # Pivot to matrix: VLMs × Questions (average bias per question)
    heatmap_data = bias_df.pivot_table(
        index="AGENT",
        columns="QUESTION_NUM",
        values="avg_bias_sign",
        aggfunc="mean"
    )

    if heatmap_data.empty:
        print("  ⚠ No data available for heatmap.")
        return

    # Keep consistent VLM order across all plots (alphabetical)
    vlm_order = sorted(heatmap_data.index)
    heatmap_data = heatmap_data.loc[vlm_order]

    # Add row averages (average bias across all questions)
    row_means = heatmap_data.mean(axis=1).to_frame(name="Average rating")

    # Adjust figure width based on number of questions
    num_questions = len(heatmap_data.columns)
    proportional_width = [1, num_questions]
    fig_width = max(10, min(20, num_questions * 1.5))
    cmap = get_region_colormap(video_region)

    fig, (ax_avg, ax_main) = plt.subplots(1, 2, figsize=(fig_width, 8), sharey=True, gridspec_kw={'width_ratios': proportional_width})

    # Plot row averages heatmap
    sns.heatmap(
        row_means,
        ax=ax_avg,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        yticklabels=True,
        cbar=False,
        linecolor="gray"
    )
    ax_avg.tick_params(left=False, labelleft=False)
    ax_avg.tick_params(right=True, labelright=True)
    ax_avg.tick_params(axis='y', pad=12)
    ax_avg.set_yticklabels(ax_avg.get_yticklabels(), rotation=0, fontsize=10)
    

    #add a little pad between the two heatmaps
    #add it
    plt.subplots_adjust(wspace=0.8)

    # Plot main heatmap (VLMs × Questions)
    sns.heatmap(
        heatmap_data,
        ax=ax_main,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        yticklabels=True,
        square=False,
        linewidths=0.5,
        cbar_kws={
            "label": "Avg Bias Sign\n(-1=underestimate, +1=overestimate)",
            "shrink": 1
        },
        linecolor="gray"
    )
    ax_main.set_ylabel("VLM Agent", fontsize=12)

    human_label = "Lima" if human_region == "lima" else "NYC"
    ax_main.set_title(
        f"Average Bias per Question\n{human_label} Humans",
        fontsize=config.PLOT_CONFIG["title_fontsize"],
        pad=-30,
        fontweight='bold',
    )
    ax_main.set_xlabel("Question Number", fontsize=12)
    ax_main.set_ylabel("", fontsize=12)

    plt.xticks(rotation=45, ha="right", fontsize=8)

    # Save the heatmap
    output_path = os.path.join(output_dir, f"bias_heatmap_avg_H{human_region}_V{video_region}.png")
    
    #plt.tight_layout(pad=-2.5)
    utils.ensure_output_dir(output_dir)
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {os.path.basename(output_path)}")

def plot_bias_score_distributions(bias_df_lima: pd.DataFrame, bias_df_nyc: pd.DataFrame, output_dir: str) -> None:
    """
    Plot distribution of bias scores for Lima and NYC human regions.

    Args:
        bias_df_lima: DataFrame with bias scores for Lima humans
        bias_df_nyc: DataFrame with bias scores for NYC humans
        output_dir: Directory to save plots
    """
    print("\n=== Plotting Bias Score Distributions ===")
    
    combined = pd.concat([
        bias_df_lima.assign(human_region="Lima"),
        bias_df_nyc.assign(human_region="NYC")
    ], ignore_index=True)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="human_region", y="avg_bias_sign", data=combined, palette=[config.PLOT_COLORS["HUMAN_LIMA"], config.PLOT_COLORS["HUMAN_NYC"]])
    plt.title("Distribution of Average Bias Signs by Human Region", fontsize=config.PLOT_CONFIG["title_fontsize"], fontweight='bold')
    plt.xlabel("Human Region", fontsize=12)
    plt.ylabel("Average Bias Sign\n(-1=underestimate, +1=overestimate)", fontsize=12)
    
    output_path = os.path.join(output_dir, "bias_score_distribution.png")
    utils.ensure_output_dir(output_dir)
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")

def main():
    # After all individual plots, create unified grid
    #plot_all_heatmaps_grid(config.OUTPUT_BIAS_DIR)
    """
    Main execution: compute and visualize bias analysis.
    
    Focus: Block 2 (Q6-Q10) ratings only.
    Method: Unit consensus - compare each VLM vs each human, average signs.
    Video regions: Lima (1-100), NYC (101-200).
    Output: 20 heatmaps (4 human×video combinations × 5 questions) + distribution plots.
    
    Supports lightweight caching: bias scores are saved to CSV so that
    re-runs with the same agents skip computation.
    """
    print("=" * 60)
    print("BIAS ANALYSIS - Block 2 (Q6-Q10)")
    print("Unit Consensus Method")
    print("=" * 60)
    
    # Load answers
    df = utils.load_answers()
    
    # Extract ratings (Block 2 only: Q6-Q10)
    df_ratings = extract_ratings(df)
    
    if len(df_ratings) == 0:
        print("✗ No ratings found in Block 2. Cannot perform bias analysis.")
        return
    
    print(f"\n✓ Found {len(df_ratings)} ratings in Block 2")
    print(f"  Videos: {df_ratings['VIDEO'].nunique()}")
    print(f"  Questions: {sorted(df_ratings['QUESTION_NUM'].unique())}")
    
    cache_path = config.BIAS_CACHE["scores"]
    
    # Check cache
    cached_bias = None
    if os.path.exists(cache_path):
        cached_bias = pd.read_csv(cache_path)
        cached_agents = set(cached_bias["AGENT"].unique())
        current_vlms = set(config.VLM_AGENTS)
        if cached_agents == current_vlms:
            print(f"✓ Bias scores cached ({len(cached_bias)} rows) — skipping computation")
        else:
            new_vlms = current_vlms - cached_agents
            print(f"⚡ {len(new_vlms)} new VLM(s) detected — recomputing bias")
            cached_bias = None
    
    if cached_bias is None:
        all_bias = []
        
        bias_df_lima = compute_unit_consensus_bias(df_ratings, human_region="lima", video_region="both")
        bias_df_lima["human_region"] = "lima"
        all_bias.append(bias_df_lima)
        
        bias_df_nyc = compute_unit_consensus_bias(df_ratings, human_region="nyc", video_region="both")
        bias_df_nyc["human_region"] = "nyc"
        all_bias.append(bias_df_nyc)
        
        combined = pd.concat(all_bias, ignore_index=True)
        utils.ensure_output_dir(os.path.dirname(cache_path))
        combined.to_csv(cache_path, index=False)
        print(f"✓ Cached bias scores: {cache_path}")
    else:
        # Use cached data for plotting
        bias_df_lima = cached_bias[cached_bias["human_region"] == "lima"]
        bias_df_nyc = cached_bias[cached_bias["human_region"] == "nyc"]
        
        # for q in [6, 7, 8, 9, 10]:
        #     plot_bias_heatmap_per_question(bias_df_lima, human_region="lima", video_region="both", question_num=q, output_dir=config.OUTPUT_BIAS_DIR)
        # for q in [6, 7, 8, 9, 10]:
        #     plot_bias_heatmap_per_question(bias_df_nyc, human_region="nyc", video_region="both", question_num=q, output_dir=config.OUTPUT_BIAS_DIR)
        
        # # Generate average heatmaps per question
        # plot_bias_heatmap_avg_per_question(bias_df_lima, human_region="lima", video_region="both", output_dir=config.OUTPUT_BIAS_DIR)
        # plot_bias_heatmap_avg_per_question(bias_df_nyc, human_region="nyc", video_region="both", output_dir=config.OUTPUT_BIAS_DIR)
    
    # Plot
    # for q in [6, 7, 8, 9, 10]:
    #     plot_bias_heatmap_per_question(bias_df_lima, human_region="lima", video_region="both", question_num=q, output_dir=config.OUTPUT_BIAS_DIR)
    # for q in [6, 7, 8, 9, 10]:
    #     plot_bias_heatmap_per_question(bias_df_nyc, human_region="nyc", video_region="both", question_num=q, output_dir=config.OUTPUT_BIAS_DIR)
    
    # Generate average heatmaps per question
    #PRINT THE quantity of answers by vlm 
    print("\nAnswer counts by VLM:")
    print(bias_df_lima.groupby("AGENT")["num_comparisons"].sum())
    print(bias_df_nyc.groupby("AGENT")["num_comparisons"].sum())
    
    plot_bias_heatmap_avg_per_question(bias_df_lima, human_region="lima", video_region="both", output_dir=config.OUTPUT_BIAS_DIR)
    plot_bias_heatmap_avg_per_question(bias_df_nyc, human_region="nyc", video_region="both", output_dir=config.OUTPUT_BIAS_DIR)
    
    # Combine Lima and NYC heatmaps for each question
    #for q in [6, 7, 8, 9, 10]:
    #    combine_heatmaps_to_single_png(q, config.OUTPUT_BIAS_DIR)
    
    #show distribution of bias scores for lima and nyc
    plot_bias_score_distributions(bias_df_lima, bias_df_nyc, config.OUTPUT_BIAS_DIR)
    
    
    print("\n" + "=" * 60)
    print("✓ Bias analysis complete!")
    #print("  Generated:")
    #print("    - 20 heatmaps (4 combinations × 5 questions)")
    #print("      • Lima Humans × Lima Videos (1-100)")
    #print("      • Lima Humans × NYC Videos (101-200)")
    #print("      • NYC Humans × Lima Videos (1-100)")
    #print("      • NYC Humans × NYC Videos (101-200)")
    #print("    - 8 distribution plots (4 combinations × 2 types)")
    #print("=" * 60)


if __name__ == "__main__":
    main()
