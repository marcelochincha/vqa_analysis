"""Shared utility functions for VQA analysis."""
import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from src import config


def ensure_output_dir(path: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================================
# Text normalization (shared across metrics)
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize answer text for deduplication and cache keying.
    
    Strips whitespace, collapses spaces, optionally lowercases.
    Used by SMATCH and STSB for text-pair caching.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    if not config.SMATCH_CONFIG.get("case_sensitive", True):
        text = text.lower()
    return text


# ============================================================================
# Incremental agent detection
# ============================================================================

def detect_agents_in_csv(csv_path: str = None) -> Dict[str, List[str]]:
    """Discover all agents present in the input CSV.
    
    Returns dict with keys: lima, nyc, vlm, human, all.
    Classification is by prefix: human_lima_*, human_nyc_*, else VLM.
    """
    csv_path = csv_path or config.INPUT_CSV
    df = pd.read_csv(csv_path, usecols=["AGENT"])
    agents = sorted(df["AGENT"].unique().tolist())
    
    lima = [a for a in agents if a.startswith("human_lima_")]
    nyc = [a for a in agents if a.startswith("human_nyc_")]
    human = lima + nyc
    vlm = [a for a in agents if a not in human]
    
    return {
        "lima": lima,
        "nyc": nyc,
        "human": human,
        "vlm": vlm,
        "all": human + vlm,
    }


def detect_new_agents(pairwise_csv_path: str, current_csv_path: str = None) -> Set[str]:
    """Compare agents in an existing pairwise CSV vs the current input CSV.
    
    Returns the set of agents present in the input CSV but missing from
    the pairwise results.  If the pairwise file does not exist, returns
    ALL agents from the input CSV.
    """
    current = detect_agents_in_csv(current_csv_path)
    current_agents = set(current["all"])
    
    if not os.path.exists(pairwise_csv_path):
        return current_agents
    
    pw = pd.read_csv(pairwise_csv_path, usecols=["AGENT_I", "AGENT_J"])
    existing_agents = set(pw["AGENT_I"].unique()) | set(pw["AGENT_J"].unique())
    
    new = current_agents - existing_agents
    if new:
        print(f"  ⚡ Detected {len(new)} new agent(s): {sorted(new)}")
    return new


def get_missing_pairs(existing_pairwise_df: pd.DataFrame | None, all_agents: List[str]) -> Set[Tuple[str, str]]:
    """Return the set of (agent_i, agent_j) pairs not yet in existing_pairwise_df.
    
    Pairs are stored as sorted tuples so (A,B) == (B,A).
    """
    from itertools import combinations
    all_pairs = {tuple(sorted(p)) for p in combinations(all_agents, 2)}
    
    if existing_pairwise_df is None or existing_pairwise_df.empty:
        return all_pairs
    
    existing = set()
    for _, row in existing_pairwise_df.iterrows():
        existing.add(tuple(sorted([row["AGENT_I"], row["AGENT_J"]])))
    
    missing = all_pairs - existing
    return missing


def load_answers(csv_path: str = config.INPUT_CSV) -> pd.DataFrame:
    """Load answers CSV from input data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} answers from {os.path.basename(csv_path)}")
    return df


def load_metric_scores(pairwise_path: str, aggregated_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pairwise and aggregated metric scores."""
    pairwise_df = None
    aggregated_df = None
    
    if os.path.exists(pairwise_path):
        pairwise_df = pd.read_csv(pairwise_path)
        print(f"✓ Loaded pairwise scores: {os.path.basename(pairwise_path)}")
    
    if aggregated_path and os.path.exists(aggregated_path):
        aggregated_df = pd.read_csv(aggregated_path)
        print(f"✓ Loaded aggregated scores: {os.path.basename(aggregated_path)}")
    
    return pairwise_df, aggregated_df


def get_agent_groups() -> Dict[str, List[str]]:
    """Get agent groups — auto-detected from CSV, falling back to config lists."""
    try:
        return detect_agents_in_csv()
    except Exception:
        return {
            "lima": config.LIMA_AGENTS,
            "nyc": config.NYC_AGENTS,
            "human": config.HUMAN_AGENTS,
            "vlm": config.VLM_AGENTS,
            "all": config.HUMAN_AGENTS + config.VLM_AGENTS
        }


def get_agent_color(agent_name: str) -> str:
    """Get color for an agent based on its type."""
    if agent_name in config.AGENT_COLORS_MAP:
        return config.AGENT_COLORS_MAP[agent_name]
    return "#999999"  # Default gray


def get_plot_style() -> Dict:
    """Get consistent plot styling parameters."""
    return {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": config.PLOT_CONFIG["dpi"]
    }

def compute_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Add block column to dataframe based on question numbers."""
    if "QUESTION_NUM" in df.columns:
        df["BLOCK"] = ((df["QUESTION_NUM"] - 1) // config.QUESTIONS_PER_BLOCK) + 1
    return df


def extract_video_id(video_value) -> int:
    """Extract trailing numeric ID from VIDEO value.

    Examples
    --------
    Robusto2_153 -> 153
    video_009 -> 9
    malformed -> 0
    """
    match = re.search(r"(\d+)$", str(video_value))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def map_video_region(video_id: int) -> str:
    """Map video ID to region using canonical thresholds.

    1-100 => lima
    101-200 => nyc
    else => unknown
    """
    if 1 <= video_id <= 100:
        return "lima"
    if 101 <= video_id <= 200:
        return "nyc"
    return "unknown"


def infer_video_region(video_value) -> str:
    """Infer video region label from a VIDEO field value."""
    return map_video_region(extract_video_id(video_value))


def add_video_region_columns(
    df: pd.DataFrame,
    video_col: str = "VIDEO",
    id_col: str = "VIDEO_NUM",
    region_col: str = "VIDEO_REGION",
) -> pd.DataFrame:
    """Return dataframe with standardized video id and region columns added."""
    result = df.copy()
    if video_col not in result.columns:
        result[id_col] = 0
        result[region_col] = "unknown"
        return result

    result[id_col] = result[video_col].apply(extract_video_id)
    result[region_col] = result[id_col].apply(map_video_region)
    return result


def aggregate_scores_by_block(pairwise_df: pd.DataFrame, score_column: str = "score") -> pd.DataFrame:
    """Aggregate pairwise scores by blocks.
    
    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise scores with VIDEO, QUESTION_NUM, AGENT_I, AGENT_J, and score columns.
    score_column : str, optional
        Name of the score column to aggregate (default: 'score').
    
    Returns
    -------
    pd.DataFrame
        Aggregated scores with BLOCK, AGENT_I, AGENT_J, MEAN_SCORE, COUNT, etc.
    """
    df = pairwise_df.copy()
    
    # Add block column
    compute_blocks(df)
    
    # Aggregate by block and agent pair
    aggregated = df.groupby(['BLOCK', 'AGENT_I', 'AGENT_J'])[score_column].agg([
        ('MEAN_SCORE', 'mean'),
        ('COUNT', 'count'),
        ('MIN_SCORE', 'min'),
        ('MAX_SCORE', 'max'),
        ('STD_SCORE', 'std'),
    ]).reset_index()
    
    return aggregated