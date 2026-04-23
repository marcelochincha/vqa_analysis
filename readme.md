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

Output: `outputs/pipeline/embed/pca_by_block_sector.png`

### 3. Cosine Similarity
```powershell
python -m pipeline cosine --data data/r2_cleaned.csv --embeddings external_embeds/... --progress
```

Output: `outputs/pipeline/cosine/cosine_heatmap_grid.png`

Use `--progress` to show tqdm progress bars.

### 4. RSA
```powershell
python -m pipeline rsa --data data/r2_cleaned.csv --embeddings external_embeds/... --progress
```

Output: `outputs/pipeline/rsa/rsa_heatmap_grid.png`

### 5. Bias (Violin Plots)
```powershell
python -m pipeline bias --data data/r2_cleaned.csv
```

Output: `outputs/pipeline/bias/bias_violin_distribution.png`

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

- **Block 1**: Questions 1-5 (video identification tasks)
- **Block 2**: Questions 6-10 (rating scale 1-10)

The pipeline computes metrics separately per block and per video sector (Lima/NYC).

---

## Requirements

```powershell
pip install -r requirements.txt
```

Key dependencies:
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `seaborn`, `matplotlib`
- `tqdm`