# VQA Analysis Pipeline

A modular pipeline for analyzing Visual Question Answering (VQA) data from human annotators and Vision-Language Models (VLMs). Generates cosine similarity heatmaps, RSA (Representational Similarity Analysis), bias violin plots, and PCA embeddings.

## Quick Start

```powershell
cd F:\robusto\vqa_analysis
python -m pipeline --all
```

This runs all 5 stages in sequence: `preprocess` → `embed` → `cosine` → `rsa` → `bias`

Note: this assumes you already have the preprocessed CSV and an embeddings cache. See the Full setup section below for the end-to-end workflow.

---

## Full Setup (Linux / Lambda)

### 1. Install everything (Miniconda + envs)

```bash
bash scripts/setup.sh
```

### 2. Preprocess (merge raw data into a cleaned CSV)

This step combines the human CSV and VLM JSONs and produces the cleaned CSV used downstream.

```bash
conda activate vqa-pipeline
python -m pipeline preprocess --human-csv data/raw/humans/answers_raw_human.csv --vlm-dir data/raw/vlms
```

Output: `data/r2_cleaned.csv`

### 3. Generate embeddings (run after preprocess)

Embeddings are computed from the cleaned CSV produced in the previous step.

```bash
conda activate vqa-embed
python scripts/generate_embeddings.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --data data/r2_cleaned.csv \
    --output external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl \
    --batch-size 64 \
    --resume
```

### 4. Run all comparisons (pipeline stages)

```bash
conda activate vqa-pipeline
python -m pipeline --all \
    --data data/r2_cleaned.csv \
    --embeddings external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl \
    --progress
```

### Run judge with vLLM (engine + judge)

Run vLLM in one terminal:

```bash
conda activate vqa-vllm
bash scripts/bash.sh
```

Then run judge in another terminal:

```bash
bash run_judge.sh
```

You can override model and base URL via env vars:

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=Qwen/Qwen3-4B \
bash run_judge.sh
```

### Optional: single command for embeddings + pipeline

If `data/r2_cleaned.csv` already exists, you can run the simple script:

```bash
bash scripts/run_all.sh
```

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
Run this after preprocess in a separate conda environment (GPU-friendly). This writes the keyed pickle used by the pipeline.

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
| `--clear-cache` | Delete cached intermediate parquet files before running. With no stages given, clears all and exits. | False |

---

## Caching & Resumability

Heavy stages save their intermediate results to parquet so that subsequent runs (or restarts after a crash) skip the expensive recomputation and just re-plot.

| Stage | Cache file (under `outputs/pipeline/<stage>/`) |
|---|---|
| `embed` | `pca_coords.parquet` |
| `cosine` | `cosine_similarity_data.parquet` |
| `rsa` | `rsa_correlations.parquet` |
| `judge` | `llm_agreement_scores.parquet` (row-level resume during the LLM loop) |

All writes are atomic (`.tmp` + rename), so a crash mid-save never leaves a corrupt parquet behind.

### Clearing the cache

```powershell
# Clear every stage's cache and exit
python -m pipeline --clear-cache

# Clear only specific stages, then re-run them from scratch
python -m pipeline --clear-cache cosine rsa

# Clear everything, then run the full pipeline fresh
python -m pipeline --clear-cache --all
```

`preprocess` and `bias` have no cache — `preprocess`'s output `data/r2_cleaned.csv` already acts as cache for downstream stages, and `bias` is fast enough to redo.

---

## Output Files

```
outputs/pipeline/
├── embed/
│   ├── pca_by_block_sector.png
│   └── pca_coords.parquet            # cache
├── cosine/
│   ├── cosine_heatmap_grid.png
│   └── cosine_similarity_data.parquet  # cache
├── rsa/
│   ├── rsa_heatmap_grid.png
│   └── rsa_correlations.parquet      # cache
├── bias/
│   └── bias_violin_distribution.png
└── judge/
    ├── judge_scores.png
    ├── llm_agreement_scores.parquet  # cache (row-level resume)
    ├── comparable_pairs.parquet
    ├── non_comparable_pairs.parquet
    ├── comparable_pairs.csv
    ├── non_comparable_pairs.csv
    └── summary.json
```

---

## Agent Ordering Convention

All heatmaps and violin plots use the same agent ordering, defined in [`pipeline/utils/metrics.py:get_ordered_agents`](pipeline/utils/metrics.py):

1. **HUMAN_LIMA** (human annotators from Lima)
2. **HUMAN_NYC** (human annotators from NYC)
3. **VLMs**

Each group is sorted **naturally** (so `human_lima_2` comes before `human_lima_17`, not after). This single helper is consumed by every plotting stage (`cosine`, `rsa`, `judge`, `bias`) so the order is identical across every figure.

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