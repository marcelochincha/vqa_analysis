"""Shared loaders for the rebuttal statistical analyses.

Everything is read-only over the original artifacts. Nothing here re-runs a model.
All functions return tidy pandas frames annotated with the 2x3 factorial factors:

    video_geo   in {Lima, NYC}          (property of the clip)
    system_type in {Human_Lima, Human_NYC, VLM}  (property of the annotator)

The Lima/NYC split of the 20 study clips is taken from the judge parquet's
VIDEO_SECTOR column and is a clean 10/10 (Lima = id<=82, NYC = id>=113).
"""
from __future__ import annotations

import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

DATA_CSV = os.path.join(REPO, "data", "r2_cleaned.csv")
EMBED_MPNET = os.path.join(
    REPO, "external_embeds", "allmpnet_batch1_r2_embeddings_cache_keyed.pkl"
)
EMBED_QWEN = os.path.join(
    REPO, "external_embeds", "qwen3emb4b_batch1_r2_embeddings_cache_keyed.pkl"
)
JUDGE_PARQUET = os.path.join(
    REPO, "outputs", "pipeline_non_embed", "judge", "llm_agreement_scores.parquet"
)

OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- constants
LIMA_VIDEOS = [
    "Robusto2_18", "Robusto2_22", "Robusto2_37", "Robusto2_40", "Robusto2_46",
    "Robusto2_56", "Robusto2_58", "Robusto2_62", "Robusto2_69", "Robusto2_82",
]
NYC_VIDEOS = [
    "Robusto2_113", "Robusto2_115", "Robusto2_131", "Robusto2_153", "Robusto2_171",
    "Robusto2_180", "Robusto2_181", "Robusto2_182", "Robusto2_192", "Robusto2_195",
]
VIDEO_GEO = {v: "Lima" for v in LIMA_VIDEOS} | {v: "NYC" for v in NYC_VIDEOS}

BLOCK_NAME = {1: "Factual", 2: "Ratings", 3: "Counterfactual", 4: "Reasoning"}

# Inference-budget groups (Supp Table 3): closed Gemini @1fps T=1.0 vs open @10fps T=0.5
CLOSED_VLMS = {"Gemini3-Flash-preview", "Gemini3-Pro-preview"}
# Systems sharing a family with the Qwen3-4B judge / Qwen3-Embedding-4B (for #10 ablation)
QWEN_FAMILY = {"Qwen3-VL-8B-Instruct", "Cosmos-Reason2-8B"}


def system_type(agent: str) -> str:
    a = agent.lower()
    if a.startswith("human_lima"):
        return "Human_Lima"
    if a.startswith("human_nyc"):
        return "Human_NYC"
    return "VLM"


def is_vlm(agent: str) -> bool:
    return system_type(agent) == "VLM"


# ---------------------------------------------------------------- loaders
@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """Full cleaned answers table with factorial factors attached."""
    d = pd.read_csv(DATA_CSV)
    d["video_geo"] = d["VIDEO"].map(VIDEO_GEO)
    d["system_type"] = d["AGENT"].map(system_type)
    d["block_name"] = d["BLOCK"].map(BLOCK_NAME)
    return d


def load_ratings(rep: int | None = 1) -> pd.DataFrame:
    """Block-2 numeric ratings (Q6-Q10), one tidy row per (agent, video, question).

    ANSWER is already normalised to an integer in [1,10] by the preprocess step
    (NaN where unparseable). Pass rep=None to keep all repetitions (VLMs have 20,
    humans have 1); rep=1 gives the symmetric single-sample view used in the paper.
    """
    d = load_raw()
    d = d[d["BLOCK"] == 2].copy()
    if rep is not None:
        d = d[d["REPETITION"] == rep].copy()
    d["rating"] = pd.to_numeric(d["ANSWER"], errors="coerce")
    d = d.dropna(subset=["rating", "video_geo"])
    d["rating"] = d["rating"].astype(int)
    return d.reset_index(drop=True)


@lru_cache(maxsize=2)
def load_embeddings(which: str = "mpnet") -> dict:
    """Keyed embedding cache: {(system, video, question_num, repetition): vector}."""
    path = EMBED_MPNET if which == "mpnet" else EMBED_QWEN
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return cache


def embeddings_frame(which: str = "mpnet", rep: int = 1) -> pd.DataFrame:
    """Long frame of embeddings for repetition `rep`, with factorial factors.

    Columns: system, video, question_num, block, video_geo, system_type, vec (np.ndarray).
    """
    cache = load_embeddings(which)
    rows = []
    for key, vec in cache.items():
        system, video, qnum, repetition = key
        if repetition != rep:
            continue
        if video not in VIDEO_GEO:
            continue
        qn = int(qnum)
        block = 1 if qn <= 5 else 2 if qn <= 10 else 3 if qn <= 15 else 4
        rows.append({
            "system": system,
            "video": video,
            "question_num": qn,
            "block": block,
            "video_geo": VIDEO_GEO[video],
            "system_type": system_type(system),
            "vec": np.asarray(vec, dtype=np.float64).ravel(),
        })
    return pd.DataFrame(rows)


def load_judge() -> pd.DataFrame:
    """Per-pair judge scores with block name and family/budget annotations."""
    d = pd.read_parquet(JUDGE_PARQUET)
    d["block_name"] = d["BLOCK"].map(BLOCK_NAME)
    return d


def ordered_systems() -> list[str]:
    """VLMs, then human_lima, then human_nyc (natural sort within group)."""
    import re

    def natkey(s):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

    d = load_raw()
    agents = d["AGENT"].unique().tolist()
    vlms = sorted([a for a in agents if is_vlm(a)], key=natkey)
    lima = sorted([a for a in agents if system_type(a) == "Human_Lima"], key=natkey)
    nyc = sorted([a for a in agents if system_type(a) == "Human_NYC"], key=natkey)
    return vlms + lima + nyc


if __name__ == "__main__":
    d = load_raw()
    print("raw rows:", len(d), "| agents:", d.AGENT.nunique())
    print("geo counts:", d.drop_duplicates("VIDEO").video_geo.value_counts().to_dict())
    r = load_ratings()
    print("ratings rows (rep1):", len(r),
          "| by system_type:", r.system_type.value_counts().to_dict())
    print("NaN rate ratings (rep1):",
          round(1 - len(r) / len(load_raw().query('BLOCK==2 and REPETITION==1')), 3))
