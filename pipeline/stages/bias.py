from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp, wasserstein_distance

from pipeline.config import PipelineConfig
from pipeline.style import apply_style, save_figure
from pipeline.utils.io import load_csv
from pipeline.utils.metrics import get_ordered_agents, get_video_region, to_numeric


QUESTIONS = {
    6: "Q6: Please rate the level of clutter from 1 to 10. Consider 10 as the highest level of clutter and 1 as the lowest.",
    7: "Q7: On a scale from 1 (not likely) to 10 (very likely), how likely is it that you encounter a similar driving scenario in your regular driving experience?",
    8: "Q8: On a scale from 1 (not hazardous) to 10 (very hazardous), how hazardous is the situation for the driver?",
    9: "Q9: On a scale from 1 to 10, how well do you think an autonomous vehicle would handle driving in this scene? Consider 1 as very poor driving and 10 as perfect driving.",
    10: "Q10: On a scale from 1 to 10, how well do you think you could handle driving in this scene? Consider 1 as very poor driving and 10 as perfect driving.",
}


def build_human_consensus(df_b2: pd.DataFrame) -> pd.DataFrame:
    human_mask = df_b2["AGENT"].str.contains("human", case=False, na=False)
    df_humans = df_b2[human_mask].copy()
    df_humans["HUMAN_REGION"] = df_humans["AGENT"].apply(lambda x: "NYC" if "nyc" in x.lower() else "LIMA")
    return df_humans.groupby(["QUESTION_NUM", "HUMAN_REGION", "VIDEO_REGION"], as_index=False)["ANSWER"].mean()


def compute_stats(df_vlms_r1: pd.DataFrame) -> pd.DataFrame:
    agents = df_vlms_r1["AGENT"].unique()
    questions = df_vlms_r1["QUESTION_NUM"].unique()
    stats_rows = []

    for agent in agents:
        df_agent = df_vlms_r1[df_vlms_r1["AGENT"] == agent]
        for question in questions:
            df_values = df_agent[df_agent["QUESTION_NUM"] == question][["VIDEO_REGION", "ANSWER"]]
            values = []
            for region in df_values["VIDEO_REGION"].unique():
                region_values = df_values[df_values["VIDEO_REGION"] == region]["ANSWER"]
                values.append(region_values.tolist())

            if len(values) != 2:
                continue

            r = wasserstein_distance(values[0], values[1])
            stat, p = ks_2samp(values[0], values[1])
            stats_rows.append({
                "AGENT": agent,
                "QUESTION_NUM": question,
                "WASSERSTEIN_DISTANCE": r,
                "KS_2SAMP_STATISTIC": stat,
                "KS_2SAMP_PVALUE": p,
            })

    return pd.DataFrame(stats_rows)


def run(config: PipelineConfig) -> Path:
    apply_style()
    data_path = config.resolve(config.data_file)
    outdir = config.out_path("bias")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(data_path, keep_default_na=False)
    df["REPETITION"] = df["REPETITION"].astype(np.uint8)
    df["QUESTION_NUM"] = df["QUESTION_NUM"].astype(np.uint8)
    df["BLOCK"] = df["BLOCK"].astype(np.uint8)

    df_b2 = df[df["BLOCK"] == 2].copy()
    df_b2["ANSWER"] = df_b2["ANSWER"].apply(to_numeric)
    df_b2["VIDEO_REGION"] = df_b2["VIDEO"].apply(get_video_region)

    human_consensus = build_human_consensus(df_b2)
    df_vlms = df_b2[~df_b2["AGENT"].str.contains("human", case=False, na=False)].copy()
    df_vlms_r1 = df_vlms[df_vlms["REPETITION"] == 1].copy()

    stats_df = compute_stats(df_vlms_r1)
    wasserstein_avg = stats_df.groupby("QUESTION_NUM")["WASSERSTEIN_DISTANCE"].mean().reset_index()

    agent_order = get_ordered_agents(df_vlms_r1["AGENT"].unique())

    sns.color_palette("deep")
    g = sns.FacetGrid(
        df_vlms_r1,
        row="VIDEO_REGION",
        col="QUESTION_NUM",
        hue="AGENT",
        hue_order=agent_order,
        height=3,
        aspect=1.2,
        margin_titles=True,
        sharey=True,
        sharex=True,
    )

    g.map_dataframe(
        sns.violinplot,
        x="AGENT",
        y="ANSWER",
        order=agent_order,
        inner="quart",
        cut=0,
        dodge=False,
        legend=False,
        edgecolor="gray",
        saturation=1,
        alpha=0.7,
    )

    for ax in g.axes.flat:
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.tick_params(bottom=False)
        ax.grid(True, alpha=0.3)
        ax.set_yticks(range(1, 11)) # Set y-ticks from 1 to 10

    plt.tight_layout()
    g.set_titles(row_template="{row_name}", col_template="Q{col_name}", fontweight="bold")
    g.set_axis_labels("", "Answer")
    g.figure.suptitle(
        "Block 2 answers of VLMs Compared to Human Consensus of NYC and LIMA",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )

    regs = list(df_vlms_r1["VIDEO_REGION"].unique())
    q_nums = list(df_vlms_r1["QUESTION_NUM"].unique())

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.axes[i, j]
            region = regs[i]
            question = q_nums[j]

            lima_human = human_consensus[
                (human_consensus["HUMAN_REGION"] == "LIMA")
                & (human_consensus["QUESTION_NUM"] == question)
                & (human_consensus["VIDEO_REGION"] == region)
            ]["ANSWER"].values

            nyc_human = human_consensus[
                (human_consensus["HUMAN_REGION"] == "NYC")
                & (human_consensus["QUESTION_NUM"] == question)
                & (human_consensus["VIDEO_REGION"] == region)
            ]["ANSWER"].values

            if len(lima_human):
                ax.axhline(lima_human[0], linestyle="--", linewidth=1, color="red", alpha=0.7)

            if len(nyc_human):
                ax.axhline(nyc_human[0], linestyle="--", linewidth=1, color="blue", alpha=0.7)

    g.add_legend(title="Agents")

    custom_lines = [
        Line2D([0], [0], color="red", linestyle="--", label="LIMA Human Consensus"),
        Line2D([0], [0], color="blue", linestyle="--", label="NYC Human Consensus"),
    ]

    handles, labels = g.axes[0, 0].get_legend_handles_labels()
    handles += custom_lines
    labels += ["LIMA Human Consensus", "NYC Human Consensus"]

    g._legend.remove()
    g.figure.legend(
        handles,
        labels,
        loc="center right",
        ncol=1,
        title="VLMs & Consensus",
    )

    for j, col_name in enumerate(g.col_names):
        ax = g.axes[-1, j]
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2

        raw_text = QUESTIONS.get(col_name, "")
        wrapped_text = textwrap.fill(raw_text, width=42)

        avg_wasserstein = wasserstein_avg[wasserstein_avg["QUESTION_NUM"] == col_name]["WASSERSTEIN_DISTANCE"].values[0]
        wrapped_text_2 = textwrap.fill(f"Avg. Wasserstein Distance between regions: {avg_wasserstein:.2f}", width=40)

        g.figure.text(
            x_center,
            bbox.y0 - 0.06,
            wrapped_text,
            ha="center",
            va="top",
            fontstyle="italic",
            fontsize=11,
        )

        g.figure.text(
            x_center,
            bbox.y0 - 0.22,
            wrapped_text_2,
            ha="center",
            va="top",
            fontsize=10,
            color="gray",
            fontweight="bold",
        )

    out_path = outdir / "bias_violin_distribution.png"
    save_figure(g.figure, out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return out_path