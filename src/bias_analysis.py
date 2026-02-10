def plot_all_heatmaps_grid(output_dir: str):
    """
    Create a 4x5 grid plot of all heatmaps.
    X axis: Questions Q6-Q10
    Y axis: Human × Video origin combinations
    Each cell: corresponding heatmap
    """
    import matplotlib.image as mpimg
    from matplotlib import pyplot as plt
    import numpy as np

    combinations = [
        ("lima", "lima"),   # Lima humans, Lima videos (1-100)
        ("lima", "nyc"),    # Lima humans, NYC videos (101-200)
        ("nyc", "lima"),    # NYC humans, Lima videos (1-100)
        ("nyc", "nyc")      # NYC humans, NYC videos (101-200)
    ]
    questions = [6, 7, 8, 9, 10]

    fig, axes = plt.subplots(4, 5, figsize=(40, 32), dpi=400)
    for row, (human_region, video_region) in enumerate(combinations):
        for col, q in enumerate(questions):
            ax = axes[row, col]
            fname = f"bias_heatmap_H{human_region}_V{video_region}_Q{q}.png"
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                img = mpimg.imread(fpath)
                ax.imshow(img, aspect='auto')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=32, color='red')
                ax.axis('off')
            if row == 0:
                ax.set_title(f"Q{q}", fontsize=36, fontweight="bold", pad=20)
            if col == 0:
                label = f"{human_region.upper()} H × {video_region.upper()} V"
                ax.set_ylabel(label, fontsize=32, fontweight="bold", labelpad=20)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.05, wspace=0.08, hspace=0.08)
    grid_path = os.path.join(output_dir, "bias_heatmap_grid_all.png")
    plt.savefig(grid_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved unified grid heatmap: {os.path.basename(grid_path)}")
def combine_heatmaps_to_single_png(question_num: int, output_dir: str):
    """
    Combine Lima and NYC heatmaps for a question into a single .png file.
    Assumes individual heatmaps are already saved as .png files.
    """
    import matplotlib.image as mpimg
    from matplotlib import pyplot as plt
    import numpy as np

    # Paths for individual heatmaps
    lima_path = os.path.join(output_dir, f"bias_heatmap_Hlima_Vlima_Q{question_num}.png")
    nyc_path = os.path.join(output_dir, f"bias_heatmap_Hnyc_Vnyc_Q{question_num}.png")

    # Check if both files exist
    if not (os.path.exists(lima_path) and os.path.exists(nyc_path)):
        print(f"  ⚠ Cannot combine: missing heatmap(s) for Q{question_num}")
        return

    # Load images
    img_lima = mpimg.imread(lima_path)
    img_nyc = mpimg.imread(nyc_path)

    # Combine vertically
    combined_img = np.vstack([img_lima, img_nyc])

    # Plot and save
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(combined_img)
    ax.axis('off')
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"bias_heatmap_combined_Q{question_num}.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved combined heatmap: {os.path.basename(combined_path)}")
"""Bias analysis comparing VLM ratings against human ratings by region.

Focus: Block 2 (Q6-Q10) which contains numerical ratings.
Method: Unit consensus - compare each VLM against each human individually,
        get sign (+1, -1, 0), then average signs.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from src import config, utils


def extract_video_id(video_str: str) -> int:
    """Extract numeric ID from video string (e.g., 'Robusto2_153' -> 153)."""
    match = re.search(r'(\d+)$', str(video_str))
    if match:
        return int(match.group(1))
    return 0


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
    
    # Extract video ID and add video_region column
    df_block2["video_id"] = df_block2["VIDEO"].apply(extract_video_id)
    df_block2["video_region"] = df_block2["video_id"].apply(
        lambda x: "lima" if 1 <= x <= 100 else "nyc" if 101 <= x <= 200 else "unknown"
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
        video_region: "lima" or "nyc" (which video set: 1-100 or 101-200)
    
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
    """Get colormap for region: Lima=white→red, NYC=white→blue."""
    if region in ["lima"]:
        # White to Red
        colors = ["white", "#ff6b6b", "#ee0000"]
        return LinearSegmentedColormap.from_list("lima_cmap", colors, N=256)
    elif region == "nyc":
        # White to Blue
        colors = ["white", "#4dabf7", "#0066cc"]
        return LinearSegmentedColormap.from_list("nyc_cmap", colors, N=256)
    else:
        # No colormap for other groups
        return None


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
    
    # Keep consistent VLM order across all plots (alphabetical)
    vlm_order = sorted(heatmap_data.index)
    heatmap_data = heatmap_data.loc[vlm_order]
    
    # Adjust figure width based on number of videos
    num_videos = len(heatmap_data.columns)
    fig_width = max(10, min(20, num_videos * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    cmap = get_region_colormap(video_region)
    
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={
            "label": "Avg Bias Sign\n(-1=underestimate, +1=overestimate)",
            "shrink": 0.8
        },
        linecolor="black",
        ax=ax
    )
    
    human_label = "Lima" if human_region == "lima" else "NYC"
    video_label = "Lima (1-100)" if video_region == "lima" else "NYC (101-200)"
    ax.set_title(
        f"Q{question_num}: VLM Bias vs {human_label} Humans | {video_label} Videos\n"
        f"White→{'Red' if human_region == 'lima' else 'Blue'}: Overestimation intensity",
        fontsize=14,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Video", fontsize=12)
    ax.set_ylabel("VLM Agent", fontsize=12)
    
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"bias_heatmap_H{human_region}_V{video_region}_Q{question_num}.png")
    utils.ensure_output_dir(output_dir)
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")


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


def main():
    # After all individual plots, create unified grid
    plot_all_heatmaps_grid(config.OUTPUT_BIAS_DIR)
    """
    Main execution: compute and visualize bias analysis.
    
    Focus: Block 2 (Q6-Q10) ratings only.
    Method: Unit consensus - compare each VLM vs each human, average signs.
    Video regions: Lima (1-100), NYC (101-200).
    Output: 20 heatmaps (4 human×video combinations × 5 questions) + distribution plots.
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
    
    # Analyze by 4 combinations: human region × video region
    combinations = [
        ("lima", "lima"),   # Lima humans, Lima videos (1-100)
        ("lima", "nyc"),    # Lima humans, NYC videos (101-200)
        ("nyc", "lima"),    # NYC humans, Lima videos (1-100)
        ("nyc", "nyc")      # NYC humans, NYC videos (101-200)
    ]
    
    for human_region, video_region in combinations:
        print("\n" + "=" * 60)
        print(f"Processing: {human_region.upper()} Humans × {video_region.upper()} Videos")
        print("=" * 60)
        
        # Compute unit consensus bias
        bias_df = compute_unit_consensus_bias(df_ratings, human_region=human_region, video_region=video_region)
        
        if len(bias_df) == 0:
            print(f"  ⚠ No data for this combination")
            continue
        
        # Generate heatmap for each question (Q6-Q10)
        print(f"\n=== Generating Heatmaps (H:{human_region.upper()}, V:{video_region.upper()}) ===")
        for q in [6, 7, 8, 9, 10]:
            plot_bias_heatmap_per_question(
                bias_df, 
                human_region=human_region,
                video_region=video_region,
                question_num=q, 
                output_dir=config.OUTPUT_BIAS_DIR
            )
        
        # Generate distribution plots per combination
        plot_bias_distributions(
            bias_df, 
            region=f"{human_region}_H_x_{video_region}_V", 
            output_dir=config.OUTPUT_BIAS_DIR
        )

    # Combine Lima and NYC heatmaps for each question
    for q in [6, 7, 8, 9, 10]:
        combine_heatmaps_to_single_png(q, config.OUTPUT_BIAS_DIR)
    
    print("\n" + "=" * 60)
    print("✓ Bias analysis complete!")
    print("  Generated:")
    print("    - 20 heatmaps (4 combinations × 5 questions)")
    print("      • Lima Humans × Lima Videos (1-100)")
    print("      • Lima Humans × NYC Videos (101-200)")
    print("      • NYC Humans × Lima Videos (1-100)")
    print("      • NYC Humans × NYC Videos (101-200)")
    print("    - 8 distribution plots (4 combinations × 2 types)")
    print("=" * 60)


if __name__ == "__main__":
    main()
