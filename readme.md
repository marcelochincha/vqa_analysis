# VQA Analysis Pipeline

A modular pipeline for analyzing Visual Question Answering (VQA) data from human annotators and Vision-Language Models (VLMs). Generates cosine similarity heatmaps, RSA (Representational Similarity Analysis), bias violin plots, and PCA embeddings.

## Quick Start

```powershell
cd F:\robusto\vqa_analysis
python -m pipeline --all
```

This runs all 6 stages in sequence: `preprocess` → `embed` → `cosine` → `rsa` → `bias` → `judge`

> ⚠️ `judge` requires a vLLM server running on `--base-url`. If you don't have one up, either skip it (`python -m pipeline preprocess embed cosine rsa bias`) or start vLLM first — see [Run the LLM Judge](#run-the-llm-judge) below.

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

Outputs: `data/r2.csv` (raw, block-2 untouched) and `data/r2_cleaned.csv` (block-2 normalized — used by downstream stages).

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

### Run the LLM Judge

The `judge` stage scores pairwise agreement between every two agents' answers for the same `(VIDEO, QUESTION_NUM)`. Internally it runs **two LLM passes per pair**:

1. **Stage 1 — Comparability filter:** asks the LLM whether the two answers can be meaningfully compared. Score `1` = comparable, `0` = talking past each other.
2. **Stage 2 — Agreement score:** only for pairs that passed Stage 1, asks the LLM to score `+2 / +1 / -1 / -2` on the scale of strong agreement to direct contradiction.

The prompts (with few-shot examples) and sampling params (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `enable_thinking=true`) match the source notebooks in [src/llm_judge_tests_stage1.ipynb](src/llm_judge_tests_stage1.ipynb) and [src/llm_judge_tests_stage2.ipynb](src/llm_judge_tests_stage2.ipynb).

**Dependency:** the judge needs the questions text in [`final_questions_v3.yaml`](final_questions_v3.yaml) at the workspace root.

#### Step 1 — Start a vLLM server

In one terminal:

```bash
conda activate vqa-vllm
bash scripts/bash.sh
```

This serves `Qwen/Qwen3-4B` on `http://localhost:8000/v1` by default. Override via env vars (`MODEL`, `PORT`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `TENSOR_PARALLEL`).

#### Step 2 — Run the judge stage

The bundled helper:

```bash
bash run_judge.sh
```

Or directly via the CLI:

```powershell
python -m pipeline judge `
    --data data/r2_cleaned.csv `
    --model Qwen/Qwen3-4B `
    --base-url http://localhost:8000/v1 `
    --temperature 1.0 `
    --concurrency 256 `
    --batch-size 512 `
    --checkpoint-every 2
```

Override via env vars when using the helper:

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=Qwen/Qwen3-4B \
CONCURRENCY=256 \
BATCH_SIZE=512 \
bash run_judge.sh
```

#### Judge-specific CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | LLM served by vLLM | `Qwen/Qwen3-4B` |
| `--base-url` | OpenAI-compatible endpoint | `http://localhost:8000/v1` |
| `--api-key` | API key (vLLM ignores it, but the client requires something) | `EMPTY` |
| `--temperature` | Sampling temperature | `1.0` |
| `--max-tokens` | Max **output** tokens per call. Must be `<` vLLM's `--max-model-len` minus your longest input prompt. | `8192` |
| `--concurrency` | Async requests in flight | `16` |
| `--batch-size` | Rows scored per batch | `128` |
| `--checkpoint-every` | Save the parquet checkpoint every N batches | `10` |

#### Quick smoke test

Before launching a full ~100k-pair run, validate end-to-end with a 3-pair sample against your live vLLM:

```powershell
python scripts/smoke_test_judge.py
```

Builds a tiny dataset (1 video × 1 question × 3 agents = 3 unique pairs), runs `judge.run()` end-to-end, then asserts that the parquet was written, all stage scores landed in the valid value sets, and the model's `<think>` reasoning was captured. Prints one full row at the end so you can eyeball the comparison quality. Takes ~30 seconds.

Use it whenever you change prompts, sampling params, or the served model.

#### Resume after a crash

Judge writes `outputs/pipeline/judge/llm_agreement_scores.parquet` every `--checkpoint-every` batches (atomic save). If the process dies mid-run, just re-invoke `python -m pipeline judge ...` — it loads the checkpoint and resumes from the first row without `STAGE1_SCORE` / `STAGE2_SCORE`. **No flags needed for resume — it's automatic.**

To force a clean re-run (e.g., after changing prompts or sampling params), delete the checkpoint first:

```powershell
python -m pipeline --clear-cache judge
```

#### Judge outputs

Under `outputs/pipeline/judge/`:

- `llm_agreement_scores.parquet` — full per-pair scores (Stage 1 + Stage 2 + reasoning)
- `comparable_pairs.parquet` / `.csv` — pairs where `STAGE1_SCORE == 1`
- `non_comparable_pairs.parquet` / `.csv` — pairs where `STAGE1_SCORE == 0`
- `summary.json` — totals, comparable ratio, agreement mean/std
- `judge_scores.png` — heatmap grid (agent × agent, per block × sector)

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

Outputs **two** CSVs side by side:

| File | What's inside |
|---|---|
| `data/r2.csv` | RAW concatenation of humans + VLMs. Block-2 answers preserved in their original free-text form (e.g. `"I'd say 7 out of 10"`, `"around 4"`). Useful for manual inspection or alternate cleaning. |
| `data/r2_cleaned.csv` | Same data but with block-2 answers normalized to a single integer in `[1, 10]` via `extract_number_with_log`. **This is the file every downstream stage consumes.** |

Both share the exact same schema (`AGENT, VIDEO, BLOCK, QUESTION_NUM, REPETITION, ANSWER`).

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

### 6. Judge (LLM-as-a-Judge agreement scores)

```powershell
python -m pipeline judge --data data/r2_cleaned.csv --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B
```

Requires a running vLLM server. See [Run the LLM Judge](#run-the-llm-judge) for the full workflow, all flags, and resume behavior.

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