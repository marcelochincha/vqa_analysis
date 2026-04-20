"""Configuration settings for VQA analysis pipeline."""
import os
from pathlib import Path

# Base paths (relative to workspace root)
WORKSPACE_ROOT = Path.cwd()
print(f"Workspace root set to: {WORKSPACE_ROOT}")
INPUT_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")

# Input files
INPUT_CSV = os.path.join(INPUT_DATA_DIR, "r2_cleaned.csv")

# Output directories per metric
OUTPUT_BERT_DIR = os.path.join(OUTPUTS_DIR, "output_heatmap", "output_scores")
OUTPUT_BERT_PLOTS = os.path.join(OUTPUTS_DIR, "output_heatmap", "output_plots")

OUTPUT_SMATCH_DIR = os.path.join(OUTPUTS_DIR, "output_smatch", "output_scores")
OUTPUT_SMATCH_PLOTS = os.path.join(OUTPUTS_DIR, "output_smatch", "output_plots")

OUTPUT_STSB_DIR = os.path.join(OUTPUTS_DIR, "output_stsb_roberta", "output_scores")
OUTPUT_STSB_PLOTS = os.path.join(OUTPUTS_DIR, "output_stsb_roberta", "output_plots")

OUTPUT_BIAS_DIR = os.path.join(OUTPUTS_DIR, "output_bias")
OUTPUT_EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, "output_embeddings_cleaned_allmpnet")

PAIRWISE_COMPARATIONS_FILE = os.path.join(OUTPUT_EMBEDDINGS_DIR, "pairwise_comparations.csv")

# Metric score files
BERT_SCORES = {
    "pairwise": os.path.join(OUTPUT_BERT_DIR, "pairwise_bert_scores.csv"),
    "aggregated": os.path.join(OUTPUT_BERT_DIR, "aggregated_bert_block_scores.csv")
}

SMATCH_SCORES = {
    "amr_cache": os.path.join(OUTPUT_SMATCH_DIR, "amr_cache.csv"),  # Legacy: by (agent, video, qnum)
    "amr_cache_by_text": os.path.join(OUTPUT_SMATCH_DIR, "amr_cache_by_text.csv"),  # Optimized: by unique text
    "smatch_cache_by_pairs": os.path.join(OUTPUT_SMATCH_DIR, "smatch_cache_by_pairs.csv"),  # Optimized: by text pairs
    "pairwise": os.path.join(OUTPUT_SMATCH_DIR, "pairwise_smatch_scores.csv"),
    "pairwise_partial": os.path.join(OUTPUT_SMATCH_DIR, "pairwise_smatch_scores_partial.csv"),
    "aggregated": os.path.join(OUTPUT_SMATCH_DIR, "aggregated_smatch_block_scores.csv")
}

STSB_SCORES = {
    "pairwise": os.path.join(OUTPUT_STSB_DIR, "pairwise_stsb_scores.csv"),
    "stsb_cache_by_pairs": os.path.join(OUTPUT_STSB_DIR, "stsb_cache_by_pairs.csv"),
    "aggregated": os.path.join(OUTPUT_STSB_DIR, "aggregated_stsb_block_scores.csv")
}

# Embedding cache (keyed dict format for incremental updates)
EMBEDDING_CACHE = {
    "keyed": os.path.join(OUTPUT_EMBEDDINGS_DIR, "embeddings_cache_keyed_allmpnet.pkl"),
    "legacy_npy": os.path.join(OUTPUT_EMBEDDINGS_DIR, "embeddings_cache_allmpnet.npy"),
}

# Bias cache
BIAS_CACHE = {
    "scores": os.path.join(OUTPUT_BIAS_DIR, "bias_scores_cache.csv"),
}

# Plot parameters
PLOT_CONFIG = {
    "figsize": (12, 10),
    "dpi": 150,
    "font_scale": 1.2,
    "title_fontsize": 16,
    "legend_fontsize": 12,
    #use sns color palette for better aesthetics red for lima, blue for nyc, green for vlm
    #USE HSV COLORS FOR BETTER DISTINCTION
}
import colorsys
S = 1
V = 0.5  # 80%

_colors_hsv = {
    "BASE": (0.0, 0.0, 1.0),        # Blanco puro (Valor 0 debe ser blanco para contraste)
    "VLM": (0.0, 0.0, 0.4),        # Verde bosque profundo (S alta, V media)
    "HUMAN_LIMA": (0.0, 0.9, 0.8),  # Rojo vibrante (S alta, V alta)
    "HUMAN_NYC": (0.6, 0.8, 0.9),   # Azul eléctrico (S media, V alta)
    "OTHER": (0.0, 0.0, 0.4)        # Gris carbón (Para que destaque sobre el blanco)
}

PLOT_COLORS = {}

for key, hex_color in _colors_hsv.items():
    h, s, v = hex_color
    r, g, b = colorsys.hsv_to_rgb(h, s , v)
    PLOT_COLORS[key] = (r,g,b)
    


# Agent groups — fallback lists (auto-discovery from CSV is preferred)
# These are only used when the CSV is not available.
LIMA_AGENTS = [f"human_lima_{i}" for i in range(1, 11)]
NYC_AGENTS = ["human_nyc_11", "human_nyc_12"]
HUMAN_AGENTS = LIMA_AGENTS + NYC_AGENTS

VLM_AGENTS = [
    "Cosmos-Reason2-8B",
    "Gemini3-Flash-preview",
    "Gemini3-Pro-preview",
    "InternVL3-8B",
    "LLaVA-Video-7B-Qwen2",
    "MiniCPM-o-2_6",
    "Perception-LM-8B",
    "Phi-4-multimodal-instruct",
    "Qwen3-VL-8B-Instruct",
    "VideoLLaMA3-7B"
]

def refresh_agent_lists_from_csv():
    """Update module-level agent lists from the input CSV (call once at startup)."""
    global LIMA_AGENTS, NYC_AGENTS, HUMAN_AGENTS, VLM_AGENTS, AGENT_MARKERS_MAP
    import pandas as _pd
    if not os.path.exists(INPUT_CSV):
        return  # keep fallback
    _df = _pd.read_csv(INPUT_CSV, usecols=["AGENT"])
    _agents = sorted(_df["AGENT"].unique().tolist())
    LIMA_AGENTS = [a for a in _agents if a.startswith("human_lima_")]
    NYC_AGENTS = [a for a in _agents if a.startswith("human_nyc_")]
    HUMAN_AGENTS = LIMA_AGENTS + NYC_AGENTS
    VLM_AGENTS = [a for a in _agents if a not in HUMAN_AGENTS]
    # Rebuild marker map
    _all_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    AGENT_MARKERS_MAP.clear()
    for i, agent in enumerate(VLM_AGENTS):
        AGENT_MARKERS_MAP[agent] = _all_markers[i % len(_all_markers)]
    for i, agent in enumerate(LIMA_AGENTS):
        AGENT_MARKERS_MAP[agent] = _all_markers[i % len(_all_markers)]
    for i, agent in enumerate(NYC_AGENTS):
        AGENT_MARKERS_MAP[agent] = _all_markers[i % len(_all_markers)]

# Auto-discover on import (silent fail if CSV missing)
try:
    refresh_agent_lists_from_csv()
except Exception:
    pass

# Individual agent marker mapping
VLM_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
LIMA_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
NYC_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

AGENT_MARKERS_MAP = {}
for i, agent in enumerate(VLM_AGENTS):
    AGENT_MARKERS_MAP[agent] = VLM_MARKERS[i % len(VLM_MARKERS)]
for i, agent in enumerate(LIMA_AGENTS):
    AGENT_MARKERS_MAP[agent] = LIMA_MARKERS[i % len(LIMA_MARKERS)]
for i, agent in enumerate(NYC_AGENTS):
    AGENT_MARKERS_MAP[agent] = NYC_MARKERS[i % len(NYC_MARKERS)]

# Block configuration (5 questions per block)
QUESTIONS_PER_BLOCK = 5
NUM_BLOCKS = 4

# SMATCH-specific configuration
SMATCH_CONFIG = {
    "batch_size_amr": 5,       # GPU batch for AMR parsing (optimized for 4GB VRAM)
    "max_workers": 8,          # Thread pool for parallel SMATCH scoring
    "checkpoint_amr": 15,     # Save AMR cache every N parses
    "checkpoint_scores": 1000, # Save scores every N comparisons
    "normalize_text": True,    # Strip whitespace and collapse spaces for deduplication
    "case_sensitive": True,    # Preserve case (True) or lowercase for matching (False)
    "use_text_deduplication": True  # Use optimized text-based caching (recommended)
}

# Shared incremental settings
INCREMENTAL_CONFIG = {
    "checkpoint_every": 500,   # Save intermediate results every N items
    "embed_batch_checkpoint": 500,  # Checkpoint embedding generation every N rows
    "stsb_pair_checkpoint": 1000,   # Checkpoint STSB pair scoring every N pairs
}