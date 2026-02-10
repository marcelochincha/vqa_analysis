"""Configuration settings for VQA analysis pipeline."""
import os

# Base paths (relative to workspace root)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")

# Input files
INPUT_CSV = os.path.join(INPUT_DATA_DIR, "r2.csv")

# Output directories per metric
OUTPUT_BERT_DIR = os.path.join(OUTPUTS_DIR, "output_heatmap", "output_scores")
OUTPUT_BERT_PLOTS = os.path.join(OUTPUTS_DIR, "output_heatmap", "output_plots")

OUTPUT_SMATCH_DIR = os.path.join(OUTPUTS_DIR, "output_smatch", "output_scores")
OUTPUT_SMATCH_PLOTS = os.path.join(OUTPUTS_DIR, "output_smatch", "output_plots")

OUTPUT_STSB_DIR = os.path.join(OUTPUTS_DIR, "output_stsb_roberta", "output_scores")
OUTPUT_STSB_PLOTS = os.path.join(OUTPUTS_DIR, "output_stsb_roberta", "output_plots")

OUTPUT_BIAS_DIR = os.path.join(OUTPUTS_DIR, "output_bias")
OUTPUT_EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, "output_embeddings")

# Metric score files
BERT_SCORES = {
    "pairwise": os.path.join(OUTPUT_BERT_DIR, "pairwise_bert_scores.csv"),
    "aggregated": os.path.join(OUTPUT_BERT_DIR, "aggregated_bert_block_scores.csv")
}

SMATCH_SCORES = {
    "pairwise": os.path.join(OUTPUT_SMATCH_DIR, "pairwise_smatch_scores.csv"),
    "aggregated": os.path.join(OUTPUT_SMATCH_DIR, "aggregated_smatch_block_scores.csv")
}

STSB_SCORES = {
    "pairwise": os.path.join(OUTPUT_STSB_DIR, "pairwise_stsb_scores.csv"),
    "aggregated": os.path.join(OUTPUT_STSB_DIR, "aggregated_stsb_block_scores.csv")
}

# Plot parameters
PLOT_CONFIG = {
    "figsize": (12, 10),
    "dpi": 300,
    "font_scale": 1.2,
    "cmap_similarity": "RdYlGn",
    "cmap_agreement": "Blues",
    "cmap_bias": "RdBu_r"
}

# Agent groups
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

# Color scheme for agent types
AGENT_COLORS = {
    "lima": "#2E86AB",      # Blue
    "nyc": "#06A77D",       # Green
    "vlm": "#D62828"        # Red/Orange
}

# Block configuration (5 questions per block)
QUESTIONS_PER_BLOCK = 5
NUM_BLOCKS = 4
