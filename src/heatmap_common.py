"""Common heatmap plotting utilities."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from src import config, utils


def create_similarity_matrix(aggregated_df: pd.DataFrame, 
                              block: int,
                              score_column: str = "score") -> pd.DataFrame:
    """
    Pivot aggregated scores into agent x agent matrix for a block.
    
    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Output of aggregate_pairwise_by_block().
    block_num : int
        Block number to extract.
    score_col : str, optional
        Which score column to use (default: 'MEAN_SCORE').
    
    Returns
    -------
    pd.DataFrame
        Square matrix with agents as index and columns, scores as values.
    """
    block_data = aggregated_df[aggregated_df['BLOCK'] == block].copy()
    
    # Create symmetric matrix (both (i,j) and (j,i) pairs)
    rows = []
    for _, row in block_data.iterrows():
        rows.append({
            'from': row['AGENT_I'],
            'to': row['AGENT_J'],
            'score': row[score_column],
        })
        rows.append({
            'from': row['AGENT_J'],
            'to': row['AGENT_I'],
            'score': row[score_column],  # symmetric
        })
    
    # Add diagonal (self-scores as 1.0)
    all_agents = set()
    for row in rows:
        all_agents.add(row['from'])
        all_agents.add(row['to'])
    
    for agent in all_agents:
        rows.append({
            'from': agent,
            'to': agent,
            'score': 1.0,
        })
    
    df = pd.DataFrame(rows)
    matrix = df.pivot_table(index='from', columns='to', values='score')
    print(f"✓ Created similarity matrix for Block {block} with shape {matrix.shape}")
    return matrix

def create_agreement_matrix(pairwise_scores_df: pd.DataFrame,
                            block: int,
                            threshold: float,
                            score_column: str = "score") -> pd.DataFrame:
    
    
    min_q = block * config.QUESTIONS_PER_BLOCK - config.QUESTIONS_PER_BLOCK + 1
    max_q = block * config.QUESTIONS_PER_BLOCK
    # BLOCK1 : 1 - 5
    # BLOCK2 : 6 - 10
    # BLOCK3 : 11 - 15
    # BLOCK4 : 16 - 20   
    df = pairwise_scores_df[
        (pairwise_scores_df['QUESTION_NUM'] >= min_q) & 
        (pairwise_scores_df['QUESTION_NUM'] <= max_q)
    ].copy()
    
    all_agents = set(df['AGENT_I'].unique()) | set(df['AGENT_J'].unique())
    agents = sorted(list(all_agents))
    n_agents = len(agents)
    
    # Get total number of questions in this block
    total_questions = df['QUESTION_NUM'].nunique()
    
    # Initialize agreement matrix
    agreement_values = np.zeros((n_agents, n_agents))
    
    # For each pair of agents, calculate agreement percentage
    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            if agent_i == agent_j:
                # Self-agreement is always 100%
                agreement_values[i, j] = 100.0
            else:
                # Get scores for this pair (could be in either direction)
                pair_scores = df[
                    ((df['AGENT_I'] == agent_i) & (df['AGENT_J'] == agent_j)) |
                    ((df['AGENT_I'] == agent_j) & (df['AGENT_J'] == agent_i))
                ][score_column].values
                
                if len(pair_scores) > 0:
                    # Count questions where score >= threshold
                    agreements = np.sum(pair_scores >= threshold)
                    agreement_pct = (agreements / len(pair_scores)) * 100.0
                    agreement_values[i, j] = agreement_pct
                else:
                    # No data for this pair
                    agreement_values[i, j] = 0.0
    
    # Create DataFrame with proper labels
    agreement_matrix = pd.DataFrame(
        agreement_values,
        index=agents,
        columns=agents
    )
    return agreement_matrix

def plot_similarity_heatmap(matrix: pd.DataFrame,
                             title: str,
                             output_path: str,
                             cmap: str = "RdYlGn",
                             vmin: float = 0,
                             vmax: float = 1,
                             annot: bool = True,
                             fmt: str = ".2f") -> None:
    """Plot similarity heatmap with annotations."""
    plt.style.use("default")
    
    fig, ax = plt.subplots(figsize=config.PLOT_CONFIG["figsize"])
    
    # Create mask for upper triangle (optional)
    #mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(
        matrix,
     #   mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"shrink": 0.8, "label": "Similarity Score"},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Agent (Reference)", fontsize=12)
    ax.set_ylabel("Agent (Candidate)", fontsize=12)
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    utils.ensure_output_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")


def plot_agreement_heatmap(matrix: pd.DataFrame,
                            title: str,
                            output_path: str,
                            cmap) -> None:
    """Plot agreement heatmap with categorical coloring based on thresholds."""
    plt.style.use("default")
    
    # Create categorical matrix
    agreement_matrix = matrix.copy()
    #agreement_cat = pd.DataFrame(
    #    np.where(matrix >= thresholds["high"], 3,
    #    np.where(matrix >= thresholds["medium"], 2, 1)),
    #    index=matrix.index,
    #    columns=matrix.columns
    #)
    
    fig, ax = plt.subplots(figsize=config.PLOT_CONFIG["figsize"])
    
    # Mask upper triangle
    # mask = np.triu(np.ones_like(agreement_cat, dtype=bool), k=1)
    
    # Custom colormap for categories
    sns.heatmap(
        agreement_matrix,
    #    mask=mask,
        annot=matrix,
        fmt=".0f",
        cmap=cmap,
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "shrink": 0.8,
            "label": "Agreement Level %",
        },
        ax=ax
    )
    
    # Update colorbar labels
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Agent (Reference)", fontsize=12)
    ax.set_ylabel("Agent (Candidate)", fontsize=12)
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    utils.ensure_output_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")
