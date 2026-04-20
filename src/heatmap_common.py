"""Common heatmap plotting utilities."""
import os
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("Agg")  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
from src.old import config
import matplotlib.patches as patches

from src.old import utils


# ============================================================================
# Helper Functions for Grouped Analysis
# ============================================================================

def get_agent_group(agent: str) -> str:
    """Determine group of agent (VLM, LIMA, or NYC).
    
    Uses dynamic agent lists from config (auto-discovered from CSV).
    
    Parameters
    ----------
    agent : str
        Agent name
    
    Returns
    -------
    str
        One of: "VLM", "LIMA", "NYC", or "UNKNOWN"
    """
    if agent in config.LIMA_AGENTS:
        return "LIMA"
    elif agent in config.NYC_AGENTS:
        return "NYC"
    elif agent.startswith("human_lima_"):
        return "LIMA"
    elif agent.startswith("human_nyc_"):
        return "NYC"
    elif agent in config.VLM_AGENTS:
        return "VLM"
    else:
        # Fallback: anything not human is VLM
        if not agent.startswith("human_"):
            return "VLM"
        return "UNKNOWN"


def reorder_matrix_by_groups(matrix: pd.DataFrame) -> pd.DataFrame:
    """Reorder matrix rows and columns to group by agent type.
    
    Order: VLM agents first, then LIMA agents, then NYC agents.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Square similarity/agreement matrix with agents as index and columns
    
    Returns
    -------
    pd.DataFrame
        Reordered matrix with agents grouped by type
    """
    agents = matrix.index.tolist()
    
    vlm_agents = [a for a in agents if get_agent_group(a) == "VLM"]
    lima_agents = [a for a in agents if get_agent_group(a) == "LIMA"]
    nyc_agents = [a for a in agents if get_agent_group(a) == "NYC"]
    
    # Maintain sort within each group for consistency
    #use natural sort
    from natsort import natsorted
    vlm_agents = natsorted(vlm_agents)
    lima_agents = natsorted(lima_agents)
    nyc_agents = natsorted(nyc_agents)
    
    # Concatenate in order: VLM -> LIMA -> NYC
    ordered_agents = vlm_agents + lima_agents + nyc_agents
    
    # Reorder rows and columns 
    reordered_matrix = matrix.loc[ordered_agents, ordered_agents]
    
    #check that the 0,0 agent is the same
    print("Top-left agent before reorder:", agents[0])
    print("Top-left agent after reorder:", reordered_matrix.index[0])
        
    
    
    print(f"✓ Reordered matrix: {len(vlm_agents)} VLM, {len(lima_agents)} LIMA, {len(nyc_agents)} NYC agents")
    return reordered_matrix, (len(vlm_agents), len(lima_agents), len(nyc_agents))


def get_group_pair(agent_i: str, agent_j: str) -> str:
    """Get the group pair for two agents.
    
    Returns one of: "VLM-VLM", "LIMA-LIMA", "NYC-NYC", "INTER"
    """
    group_i = get_agent_group(agent_i)
    group_j = get_agent_group(agent_j)
    
    if group_i == group_j:
        return f"{group_i}"
    else:
        return "INTER"


def create_grouped_colormap(group_pair: str) -> LinearSegmentedColormap:
    """Create LinearSegmentedColormap for a group pair.
    
    - Same group: white to group color
    - Different groups: white to gray
    
    Parameters
    ----------
    group_pair : str
        One of: "VLM-VLM", "LIMA-LIMA", "NYC-NYC", "INTER"
    
    Returns
    -------
    LinearSegmentedColormap
        Colormap from white to endpoint color
    """
    base = config.PLOT_COLORS["BASE"]  # Base color (white or light gray)
    if group_pair == "VLM":
        # VLM: green
        vlm_color = config.PLOT_COLORS["VLM"]
        return LinearSegmentedColormap.from_list(
            "vlm_cmap",
            [base, vlm_color]
        )
    elif group_pair == "LIMA":
        # LIMA: blue
        lima_color = config.PLOT_COLORS["HUMAN_LIMA"]
        return LinearSegmentedColormap.from_list(
            "lima_cmap",
            [base, lima_color]
        )
    elif group_pair == "NYC":
        # NYC: red
        nyc_color = config.PLOT_COLORS["HUMAN_NYC"]
        return LinearSegmentedColormap.from_list(
            "nyc_cmap",
            [base, nyc_color]
        )
    else:  # INTER
        # Inter-group: white to gray
        other_color = config.PLOT_COLORS["OTHER"]
        return LinearSegmentedColormap.from_list(
            "inter_cmap",
            [base, other_color]
        )

def create_similarity_matrix(aggregated_df: pd.DataFrame,
                              block: int,
                              score_column: str = "score",
                              video_region: str = "both",
                              pairwise_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
    if video_region not in {"both", "lima", "nyc"}:
        raise ValueError("video_region must be one of: both, lima, nyc")

    working_score_column = score_column

    if video_region == "both":
        block_data = aggregated_df[aggregated_df['BLOCK'] == block].copy()
        if working_score_column not in block_data.columns:
            raise ValueError(f"Column '{working_score_column}' not found in aggregated_df")
    else:
        if pairwise_df is None:
            raise ValueError("pairwise_df is required when video_region is not 'both'")

        min_q = block * config.QUESTIONS_PER_BLOCK - config.QUESTIONS_PER_BLOCK + 1
        max_q = block * config.QUESTIONS_PER_BLOCK

        df = pairwise_df[
            (pairwise_df['QUESTION_NUM'] >= min_q) &
            (pairwise_df['QUESTION_NUM'] <= max_q)
        ].copy()

        if "VIDEO" not in df.columns:
            raise ValueError("pairwise_df must include VIDEO column for region filtering")

        df["VIDEO_REGION"] = df["VIDEO"].apply(utils.infer_video_region)
        df = df[df["VIDEO_REGION"] == video_region].copy()

        if df.empty:
            print(f"⚠ No pairwise rows for Block {block}, region={video_region}")
            return pd.DataFrame()

        if working_score_column not in df.columns:
            fallback_candidates = ["score", "BERT_SCORE", "COSINE_SCORE", "MEAN_SCORE"]
            fallback = next((c for c in fallback_candidates if c in df.columns), None)
            if fallback is None:
                raise ValueError(
                    f"No usable score column found in pairwise_df. Tried '{working_score_column}' and {fallback_candidates}"
                )
            print(
                f"⚠ Column '{working_score_column}' not found in pairwise_df; using '{fallback}' for block {block}, region={video_region}"
            )
            working_score_column = fallback

        block_data = df.groupby(["AGENT_I", "AGENT_J"], as_index=False)[working_score_column].mean()
    
    # Create symmetric matrix (both (i,j) and (j,i) pairs)
    rows = []
    for _, row in block_data.iterrows():
        rows.append({
            'from': row['AGENT_I'],
            'to': row['AGENT_J'],
            'score': row[working_score_column],
        })
        rows.append({
            'from': row['AGENT_J'],
            'to': row['AGENT_I'],
            'score': row[working_score_column],  # symmetric
        })
    
    if not rows:
        print(f"⚠ No similarity rows for Block {block}, region={video_region}")
        return pd.DataFrame()

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
    print(f"✓ Created similarity matrix for Block {block} ({video_region}) with shape {matrix.shape}")
    return matrix

def create_agreement_matrix(pairwise_scores_df: pd.DataFrame,
                            block: int,
                            threshold: float,
                            score_column: str = "score",
                            video_region: str = "both") -> pd.DataFrame:
    
    
    min_q = block * config.QUESTIONS_PER_BLOCK - config.QUESTIONS_PER_BLOCK + 1
    max_q = block * config.QUESTIONS_PER_BLOCK
    # BLOCK1 : 1 - 5
    # BLOCK2 : 6 - 10
    # BLOCK3 : 11 - 15
    # BLOCK4 : 16 - 20   
    if video_region not in {"both", "lima", "nyc"}:
        raise ValueError("video_region must be one of: both, lima, nyc")

    df = pairwise_scores_df[
        (pairwise_scores_df['QUESTION_NUM'] >= min_q) & 
        (pairwise_scores_df['QUESTION_NUM'] <= max_q)
    ].copy()

    if video_region != "both":
        if "VIDEO" not in df.columns:
            raise ValueError("pairwise_scores_df must include VIDEO column for region filtering")
        df["VIDEO_REGION"] = df["VIDEO"].apply(utils.infer_video_region)
        df = df[df["VIDEO_REGION"] == video_region].copy()

    if df.empty:
        print(f"⚠ No agreement rows for Block {block}, region={video_region}")
        return pd.DataFrame()
    
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
                             color_label: str,
                             output_path: str,
                             cmap: str = "RdYlGn",
                             vmin: float = 0,
                             vmax: float = 1,
                             annot: bool = True,
                             fmt: str = ".2f",
                             use_group_colors: bool = True) -> None:
    """Plot similarity heatmap with group-based colormaps.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Similarity matrix with agents as index and columns
    title : str
        Plot title
    output_path : str
        Path to save the figure
    cmap : str, optional
        Fallback colormap name (used if use_group_colors=False)
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    annot : bool, optional
        Whether to annotate cells with values
    fmt : str, optional
        Format string for annotations
    use_group_colors : bool, optional
        If True, apply group-specific colormaps; if False, use single cmap
    """
    plt.style.use("default")

    if matrix is None or matrix.empty:
        print(f"⚠ Skipping empty matrix plot: {title}")
        return
    
    matrix,(vlm_size,lima_size,nyc_size) = reorder_matrix_by_groups(matrix)
        
    fig, ax = plt.subplots(figsize=config.PLOT_CONFIG["figsize"])
    ax.set_aspect('equal', adjustable='box') 
    #plt.subplots_adjust(bottom=0.25, right=)  # Make space for colorbars on right

    # Normalize values to [0, 1] range
    values = matrix.values.copy()
    values_norm = (values - vmin) / (vmax - vmin)
    values_norm = np.clip(values_norm, 0, 1)
    
    agents = matrix.index.tolist()
    n_agents = len(agents)
    
    if use_group_colors:        
        # Plot using pcolormesh for full control
        im = ax.pcolormesh(
            np.arange(n_agents + 1),
            np.arange(n_agents + 1),
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )
        
        # Add annotations
        if annot:
            for i in range(n_agents):
                for j in range(n_agents):
                    if fmt == "%d":
                        text = f"{int(values[i, j])}"
                    else:
                        text = format(values[i, j], fmt)
                    #get the color for the cell from the color array
                    cell_color = im.cmap(im.norm(values[i, j]))
                    #calculate the brightness of the cell color
                    brightness = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                    text_color = 'black' if brightness > 0.5 else 'white'
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        text,
                        ha='center',
                        va='center',
                        fontsize=8,
                        color=text_color
                    )
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=color_label)
        ax.set_xlim(0, n_agents)
        ax.set_ylim(n_agents, 0)   # ← inviertes aquí directamente
        ax.set_xticks(np.arange(n_agents) + 0.5)
        ax.set_yticks(np.arange(n_agents) + 0.5)
        ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(agents, fontsize=9)

        # NOW add the color borders
        #use cum sum to get the positions of the group boundaries
        group_sizes = [vlm_size, lima_size, nyc_size]
        sizes_group = np.cumsum(group_sizes)
        group_colors = [config.PLOT_COLORS["VLM"], config.PLOT_COLORS["HUMAN_LIMA"], config.PLOT_COLORS["HUMAN_NYC"]]
        for i,boundary in enumerate(group_sizes):  # Skip the last boundary (end of matrix)
            #use the previous boundary to get the start of the group if the first use 0,0
            start = 0 if i == 0 else sizes_group[i-1]
            #print(f"Adding rectangle for group {i} from {start} to {boundary} with color {group_colors[i]}")
            rect = patches.Rectangle((start, start),boundary, boundary, edgecolor=group_colors[i], linewidth=2, fill=False,zorder=10,antialiased=False, joinstyle='miter')
            ax.add_patch(rect)
        
    else:
        # Original single-colormap approach
        sns.heatmap(
            matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={"shrink": 0.8, "label":  color_label},
            ax=ax,
        )
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Agent (Reference)", fontsize=12)
    ax.set_ylabel("Agent (Candidate)", fontsize=12)
    
    utils.ensure_output_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {os.path.basename(output_path)}")


# def plot_agreement_heatmap(matrix: pd.DataFrame,
#                             title: str,
#                             output_path: str,
#                             cmap: str = "RdYlGn",
#                             use_group_colors: bool = True) -> None:
#     """Plot agreement heatmap with group-based colormaps.
    
#     Parameters
#     ----------
#     matrix : pd.DataFrame
#         Agreement matrix with agents as index and columns (values 0-100)
#     title : str
#         Plot title
#     output_path : str
#         Path to save the figure
#     cmap : str, optional
#         Fallback colormap name (used if use_group_colors=False)
#     use_group_colors : bool, optional
#         If True, apply group-specific colormaps; if False, use single cmap
#     """
#     plt.style.use("default")

#     if matrix is None or matrix.empty:
#         print(f"⚠ Skipping empty matrix plot: {title}")
#         return
    
#     if use_group_colors:
#         matrix, _ = reorder_matrix_by_groups(matrix)
    
#     fig, ax = plt.subplots(figsize=config.PLOT_CONFIG["figsize"])
#     ax.set_aspect('equal', adjustable='box') 
#     plt.subplots_adjust(bottom=0.25, right=0.85)  # Make space for colorbars on right
    
#     # Normalize values to [0, 1] range (agreement is 0-100)
#     values = matrix.values.copy()
#     values_norm = values / 100.0
#     vmin, vmax = 0, 100
    
#     agents = matrix.index.tolist()
#     n_agents = len(agents)
    
#     if use_group_colors:
#         # Create custom color array
#         color_array = np.zeros((n_agents, n_agents, 3))  # RGB
        
#         for i in range(n_agents):
#             for j in range(n_agents):
#                 agent_i = agents[i]
#                 agent_j = agents[j]
#                 group_pair = get_group_pair(agent_i, agent_j)
#                 cmap_group = create_grouped_colormap(group_pair)
                
#                 # Normalize value (0-100 to 0-1) and get color
#                 norm_val = values_norm[i, j]
#                 rgb = cmap_group(norm_val)[:3]
#                 color_array[i, j] = rgb
        
#         # Plot using rectangles
#         for i in range(n_agents):
#             for j in range(n_agents):
#                 ax.add_patch(plt.Rectangle(
#                     (j, n_agents - i - 1),
#                     1, 1,
#                     facecolor=color_array[i, j],
#                     edgecolor='gray',
#                     linewidth=0.5
#                 ))
        
#         # Add annotations
#         for i in range(n_agents):
#             for j in range(n_agents):
#                 text = f"{values[i, j]:.0f}"
#                 ax.text(
#                     j + 0.5,
#                     n_agents - i - 0.5,
#                     text,
#                     ha='center',
#                     va='center',
#                     fontsize=8,
#                     color='black'
#                 )
        
#         from mpl_toolkits.axes_grid1 import make_axes_locatable
#         divider = make_axes_locatable(ax)
#         rampas = ['VLM', "LIMA", "NYC", "INTER"]  # Nombres de las rampas para cada grupo
#         for i, cmap_name in enumerate(rampas):
#             espacio = 0.2 if i == 0 else 0.0 
            
#             # Creamos el eje para la rampa
#             cax = divider.append_axes("right", size="3%", pad=espacio)
            
#             sm = plt.cm.ScalarMappable(cmap=create_grouped_colormap(cmap_name), norm=plt.Normalize(0, 1))
            
#             # Creamos la colorbar
#             cbar = fig.colorbar(sm, cax=cax)
            
#             # LÓGICA DE LOS TICKS:
#             if i < len(rampas) - 1:
#                 # Si no es la última rampa, quitamos los números/ticks
#                 cax.set_yticks([]) 
#             else:
#                 # Solo la última rampa lleva etiqueta general si gustas
#                 cbar.set_label('Similarity score', rotation=90, labelpad=15)
#                 cbar.ax.yaxis.label.set_size(12)
#                 #also sert to number ticks
#                 cbar.ax.tick_params(labelsize=12)
        
#         ax.set_xlim(0, n_agents)
#         ax.set_ylim(0, n_agents)
#         ax.set_xticks(np.arange(n_agents) + 0.5)
#         ax.set_yticks(np.arange(n_agents) + 0.5)
#         ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
#         #ax.set_yticklabels(reversed(agents), fontsize=9)
#         ax.invert_yaxis()
        
#     else:
#         # Original single-colormap approach
#         sns.heatmap(
#             matrix,
#             annot=matrix,
#             cmap=cmap,
#             fmt=".0f",
#             vmin=0,
#             vmax=100,
#             square=True,
#             linewidths=0.5,
#             linecolor='gray',
#             cbar_kws={
#                 "shrink": 0.8,
#                 "label": "Agreement Level %",
#             },
#             ax=ax
#         )
    
#     ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
#     ax.set_xlabel("Agent (Reference)", fontsize=12)
#     ax.set_ylabel("Agent (Candidate)", fontsize=12)
    
#     utils.ensure_output_dir(os.path.dirname(output_path))
#     plt.savefig(output_path, dpi=config.PLOT_CONFIG["dpi"], bbox_inches="tight")
#     plt.close()
    
#     print(f"✓ Saved: {os.path.basename(output_path)}")
