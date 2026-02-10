"""Shared utility functions for VQA analysis."""
import os
import pandas as pd
from typing import Dict, List, Tuple
from src import config


def ensure_output_dir(path: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


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
    """Get agent groups categorized by type."""
    return {
        "lima": config.LIMA_AGENTS,
        "nyc": config.NYC_AGENTS,
        "human": config.HUMAN_AGENTS,
        "vlm": config.VLM_AGENTS,
        "all": config.HUMAN_AGENTS + config.VLM_AGENTS
    }


def get_agent_color(agent_name: str) -> str:
    """Get color for an agent based on its type."""
    if agent_name in config.LIMA_AGENTS:
        return config.AGENT_COLORS["lima"]
    elif agent_name in config.NYC_AGENTS:
        return config.AGENT_COLORS["nyc"]
    elif agent_name in config.VLM_AGENTS:
        return config.AGENT_COLORS["vlm"]
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