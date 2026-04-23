from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=float).ravel()
    b = np.asarray(vec_b, dtype=float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(a, b) / denom)


def get_row_key(row) -> tuple:
    return (row["AGENT"], row["VIDEO"], row["QUESTION_NUM"], row["REPETITION"])


def build_row_map(df: pd.DataFrame, embeddings_cache: dict) -> dict:
    row_map = {}
    for row in df.itertuples(index=False):
        key = (row.AGENT, row.VIDEO, row.QUESTION_NUM, row.REPETITION)
        if key in embeddings_cache:
            row_map[key] = embeddings_cache[key]
        elif row.ANSWER in embeddings_cache:
            row_map[key] = embeddings_cache[row.ANSWER]
        else:
            raise KeyError(f"Embedding for key {key} (or answer text) not found in cache")
    return row_map


def assign_block(question_num: int) -> int:
    if question_num <= 5:
        return 1
    if question_num <= 10:
        return 2
    if question_num <= 15:
        return 3
    return 4


def get_video_sector(video_str: str) -> str:
    try:
        video_num = int(video_str.split("_")[1])
        return "Lima" if video_num <= 100 else "NYC"
    except (IndexError, ValueError):
        return "Unknown"


def get_video_region(video_str: str) -> str:
    try:
        video_num = int(video_str.split("_")[1])
        return "Videos of LIMA" if video_num <= 100 else "Videos of NYC"
    except (IndexError, ValueError):
        return "Unknown"


def to_numeric(answer) -> float | np.nan:
    try:
        return float(answer)
    except Exception:
        return np.nan


def compute_region_stats(df_vlms: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for agent in df_vlms["AGENT"].unique():
        df_agent = df_vlms[df_vlms["AGENT"] == agent]
        for question in df_vlms["QUESTION_NUM"].unique():
            df_values = df_agent[df_agent["QUESTION_NUM"] == question][["VIDEO_REGION", "ANSWER"]]
            region_values = [df_values[df_values["VIDEO_REGION"] == region]["ANSWER"].tolist() for region in df_values["VIDEO_REGION"].unique()]
            if len(region_values) != 2:
                continue
            try:
                stat, pvalue = ks_2samp(region_values[0], region_values[1])
            except Exception:
                stat, pvalue = np.nan, np.nan
            stats.append(
                {
                    "AGENT": agent,
                    "QUESTION_NUM": question,
                    "WASSERSTEIN_DISTANCE": wasserstein_distance(region_values[0], region_values[1]),
                    "KS_2SAMP_STATISTIC": stat,
                    "KS_2SAMP_PVALUE": pvalue,
                }
            )
    return pd.DataFrame(stats)