# VQA Analysis Pipeline

A modular pipeline for analyzing Visual Question Answering (VQA) data from human annotators and Vision-Language Models (VLMs). Generates cosine similarity heatmaps, RSA (Representational Similarity Analysis), bias violin plots, and PCA embeddings.

## Quick Start

```powershell
cd F:\robusto\vqa_analysis
python -m pipeline --all
```

This runs all 5 stages in sequence: `preprocess` → `embed` → `cosine` → `rsa` → `bias`

---

## Data Placement

### Raw Human Answers
Place your raw human survey CSV at:
```
data/raw/humans/answers_raw_human.csv
```

Expected format: Wide format with columns like `R2_153-Q1`, `R2_153-Q2`, etc. (survey question columns).

### VLM JSON Files
Place your VLM response JSON files in:
```
data/raw/vlms/
```

Each JSON file should contain an array of objects with `video`, `question`, and `response` fields.

### Embeddings Cache
Place your pre-computed embeddings cache at:
```
external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl
```

Or specify a custom path with `--embeddings`.

### Generate Embeddings Cache (separate step)
Run this in a separate conda environment (GPU-friendly). This writes the keyed pickle used by the pipeline.

```bash
conda activate <env>
python scripts/generate_embeddings.py --model sentence-transformers/all-mpnet-base-v2 --output external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl --batch-size 64 --resume

# Qwen embedding model (adjust batch size for your hardware)
python scripts/generate_embeddings.py --model Qwen/Qwen3-Embedding-4B --output external_embeds/cleaned_embeddings_cache_keyed_qwen.pkl --batch-size 1 --trust-remote-code --resume
```

Optional .npz output for downstream tooling:
```bash
python scripts/generate_embeddings.py --model sentence-transformers/all-mpnet-base-v2 --output external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl --npz external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.npz --resume
```

---

## Running Stages Individually

### 1. Preprocess (required first)
```powershell
python -m pipeline preprocess --human-csv data/raw/humans/answers_raw_human.csv --vlm-dir data/raw/vlms
```

Output: `data/r2_cleaned.csv`

### 2. Embed (PCA scatter plots)

```powershell
python -m pipeline embed --data data/r2_cleaned.csv --embeddings external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl
```

Uses **Repetition 1 only** (representative sample per agent-video-question).

### 3. Cosine Similarity

```powershell
python -m pipeline cosine --data data/r2_cleaned.csv --embeddings external_embeds/... --progress
```

Uses **Repetition 1 only** (representative sample per agent-video-question).

### 4. RSA

```powershell
python -m pipeline rsa --data data/r2_cleaned.csv --embeddings external_embeds/... --progress
```

Uses **Repetition 1 only** (representative sample per agent-video-question).

### 5. Bias (Violin Plots)
```powershell
python -m pipeline bias --data data/r2_cleaned.csv
```

Uses **Block 2, Repetition 1 only** (rating scale questions 6-10).

---

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `stages` | Stages to run (space-separated) | - |
| `--all` | Run all stages | - |
| `--list` | List available stages | - |
| `--progress` | Show progress bars | False |
| `--data` | Input CSV (after preprocess) | `data/r2_cleaned.csv` |
| `--embeddings` | Embeddings cache file | `external_embeds/...` |
| `--outdir` | Output directory | `outputs/pipeline` |
| `--human-csv` | Raw human CSV for preprocess | `data/raw/humans/answers_raw_human.csv` |
| `--vlm-dir` | VLM JSON directory | `data/raw/vlms/` |

---

## Output Files

```
outputs/pipeline/
├── embed/
│   └── pca_by_block_sector.png
├── cosine/
│   └── cosine_heatmap_grid.png
├── rsa/
│   └── rsa_heatmap_grid.png
└── bias/
    └── bias_violin_distribution.png
```

---

## Agent Ordering Convention

All heatmaps use consistent agent ordering:

1. **VLMs** (alphabetically sorted)
2. **HUMAN_LIMA** (human annotators from Lima)
3. **HUMAN_NYC** (human annotators from NYC)

This puts VLM agents first, human baselines at the end for visual comparison.

---

## Block Definitions

- **Block 1**: Questions 1-5 (Factual - video identification)
    - Questions in Block 1 vary by video and are defined in final_questions_v3.yaml.
- **Block 2**: Questions 6-10 (Ratings - scale 1-10)
- **Block 3**: Questions 11-15 (Counterfactual & Hypothetical)
- **Block 4**: Questions 16+ (Reasoning)

The pipeline computes metrics separately per block and per video sector (Lima/NYC).

> **Note**: All stages use Repetition 1 only (a single representative sample per agent-video-question).

---

## Requirements

```powershell
pip install -r requirements.txt
```

Key dependencies:
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `seaborn`, `matplotlib`
- `tqdm`