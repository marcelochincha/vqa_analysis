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

## Bash Workflow (recommended)

All operational tasks are wrapped in bash scripts you can drive entirely with environment variables. No need to touch Python args directly. Order of use:

| Step | Script | What it does |
|---|---|---|
| 1 | [`scripts/setup.sh`](scripts/setup.sh) | Install Miniconda + the three conda envs (`vqa-pipeline`, `vqa-embed`, `vqa-vllm`). |
| 2 | [`scripts/run_all.sh`](scripts/run_all.sh) | Optional preprocess → embeddings → all pipeline stages (`embed` / `cosine` / `rsa` / `bias` / `judge`). |
| 3 | [`scripts/bash.sh`](scripts/bash.sh) | Start the vLLM server that the `judge` stage talks to. **Runs in its own terminal.** |
| 4 | [`scripts/run_judge.sh`](scripts/run_judge.sh) | Run *only* the `judge` stage (handy when vLLM goes up after the other stages, or when iterating on judge params). |
| 5 | [`scripts/package_cache.sh`](scripts/package_cache.sh) | Bundle inputs + cached parquets + embeddings into a tarball for replication on another machine. |

Each script reads env vars with sane defaults. Override per invocation with `VAR=value bash script.sh`.

### 1. Install — `bash scripts/setup.sh`

One-time setup. No env vars to worry about.

### 2. Pipeline — `bash scripts/run_all.sh`

Single command that goes from raw data to plots (excluding the judge). With everything default it runs **embed → pipeline (`--all`)**. Toggle the stages on or off via env vars.

```bash
# Minimal: assumes data/r2_cleaned.csv exists and uses all-mpnet defaults
bash scripts/run_all.sh

# Full end-to-end: preprocess raw data first
RUN_PREPROCESS=true bash scripts/run_all.sh

# Re-run only the pipeline stages (you already have the embeddings cache)
SKIP_EMBED=true bash scripts/run_all.sh

# Only build embeddings (no pipeline plots yet)
SKIP_PIPELINE=true bash scripts/run_all.sh

# Switch to Qwen3-Embedding-4B (embedding output + pipeline both pick up the new path automatically)
EMBED_MODEL="Qwen/Qwen3-Embedding-4B" \
EMBED_DEVICE=cuda \
EMBED_DTYPE=bf16 \
EMBED_NORMALIZE=true \
EMBED_MAX_SEQ_LENGTH=512 \
  bash scripts/run_all.sh
```

**Workflow toggles**

| Env var | Default | Effect |
|---|---|---|
| `RUN_PREPROCESS` | `false` | Also run `python -m pipeline preprocess` before embed |
| `SKIP_EMBED` | `false` | Skip the embedding generation step |
| `SKIP_PIPELINE` | `false` | Skip the `python -m pipeline --all` step |

**Embed step env vars** (forwarded to `scripts/generate_embeddings.py`)

| Env var | Default | Maps to | Notes |
|---|---|---|---|
| `EMBED_MODEL` | `sentence-transformers/all-mpnet-base-v2` | `--model` | |
| `EMBED_BATCH_SIZE` | `1` | `--batch-size` | `1` keeps outputs reproducible across runs |
| `EMBED_OUTPUT` | *(derived)* | `--output` | If empty, both this script and the pipeline derive `external_embeds/{slug}_batch{N}_r2_embeddings_cache_keyed.pkl` from `EMBED_MODEL` + `EMBED_BATCH_SIZE` |
| `EMBED_DEVICE` | *(auto)* | `--device` | `cpu` / `cuda` / `cuda:0` |
| `EMBED_DTYPE` | *(model default)* | `--dtype` | `fp32` / `fp16` / `bf16` (use bf16 for Qwen3-Embedding-4B) |
| `EMBED_NORMALIZE` | `false` | `--normalize` | Recommended for Qwen3-Embedding |
| `EMBED_PADDING_SIDE` | *(auto)* | `--padding-side` | Auto-set to `left` for Qwen3-Embedding |
| `EMBED_INSTRUCTION` | *(none)* | `--instruction` | Optional task prefix |
| `EMBED_MAX_SEQ_LENGTH` | *(model default)* | `--max-seq-length` | `512` is plenty for VQA answers |
| `EMBED_TRUST_REMOTE_CODE` | `false` | `--trust-remote-code` | Auto-set for Qwen3-Embedding |
| `EMBED_NPZ` | *(none)* | `--npz` | Also dump a numpy `.npz` |

**Pipeline step env vars**

| Env var | Default | Effect |
|---|---|---|
| `DATA_PATH` | `data/r2_cleaned.csv` | Passed as `--data` |
| `PIPELINE_PROGRESS` | `false` | Adds `--progress` |
| `PIPELINE_OUTDIR` | `outputs/pipeline` | Passed as `--outdir` |

**Preprocess step env vars** (only used when `RUN_PREPROCESS=true`)

| Env var | Default |
|---|---|
| `HUMAN_CSV` | `data/raw/humans/answers_raw_human.csv` |
| `VLM_DIR` | `data/raw/vlms` |

### 3. Start vLLM — `bash scripts/bash.sh`

Open a separate terminal:

```bash
conda activate vqa-vllm
bash scripts/bash.sh
```

This serves `Qwen/Qwen3-4B` on `http://localhost:8000/v1` with `--reasoning-parser qwen3` enabled (so the model's `<think>...</think>` block lands in the response's `reasoning_content` field, which the judge stores).

| Env var | Default |
|---|---|
| `MODEL` | `Qwen/Qwen3-4B` |
| `HOST` | `0.0.0.0` |
| `PORT` | `8000` |
| `TENSOR_PARALLEL` | `1` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `MAX_MODEL_LEN` | `32768` |
| `DTYPE` | `auto` |
| `TRUST_REMOTE_CODE` | `false` |

### 4. Judge only — `bash scripts/run_judge.sh`

Runs *just* the `judge` stage against your live vLLM server. Resumes automatically from `outputs/pipeline/judge/llm_agreement_scores.parquet`.

```bash
bash scripts/run_judge.sh

# Smoke-test on two agents — output goes to *_test.* files so the real run isn't touched
bash scripts/run_judge.sh --agents human_lima_1 human_nyc_1

# Higher throughput, more retries
CONCURRENCY=256 BATCH_SIZE=512 MAX_RETRIES=5 bash scripts/run_judge.sh
```

| Env var | Default | Maps to |
|---|---|---|
| `DATA_PATH` | `data/r2_cleaned.csv` | `--data` |
| `MODEL` | `Qwen/Qwen3-4B` | `--model` |
| `BASE_URL` | `http://localhost:8000/v1` | `--base-url` |
| `API_KEY` | `EMPTY` | `--api-key` |
| `TEMPERATURE` | `1.0` | `--temperature` |
| `MAX_TOKENS` | `8192` | `--max-tokens` |
| `CONCURRENCY` | `16` | `--concurrency` |
| `BATCH_SIZE` | `128` | `--batch-size` |
| `CHECKPOINT_EVERY` | `10` | `--checkpoint-every` |
| `MAX_RETRIES` | `3` | `--max-retries` |

Any extra CLI flag (`--agents ...`, `--temperature ...`) appended after the script name is forwarded verbatim to `python -m pipeline judge`.

### 5. Package the cache — `bash scripts/package_cache.sh`

Bundles inputs (`data/r2_cleaned.csv`, raw sources, `final_questions_v3.yaml`), all top-level embedding caches, and every stage's parquet checkpoint into a `.tar.gz`. Reduce re-replication on another machine to: extract → run the same `bash` commands above; everything cached is reused.

```bash
bash scripts/package_cache.sh                       # default path: ./robusto_cache_<timestamp>.tar.gz
bash scripts/package_cache.sh /tmp/cache.tar.gz     # custom path
```

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
    --resume
```

Defaults to `--batch-size 1` for **reproducibility** — large embedding models can produce slightly different outputs depending on batch size (non-deterministic fp16/bf16 matmul accumulation). The output filename is derived from the model name and batch size automatically (e.g. `external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl`). Pass `--output` to override.

For **Qwen3-Embedding** family models you usually want bf16 + normalize:

```bash
python scripts/generate_embeddings.py \
    --model Qwen/Qwen3-Embedding-4B \
    --data data/r2_cleaned.csv \
    --device cuda \
    --dtype bf16 \
    --normalize \
    --max-seq-length 512 \
    --resume
```

`trust_remote_code=True` and `padding_side=left` are auto-configured when the model name contains `qwen3-embedding`.

### 4. Run all comparisons (pipeline stages)

```bash
conda activate vqa-pipeline
python -m pipeline --all \
    --data data/r2_cleaned.csv \
    --embeddings external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl \
    --progress
```

> ⚠️ **Match the `--embeddings` path to whatever you generated in step 3.** The pipeline's default `--embeddings` is `external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl`. If you used a different model or batch size when generating embeddings (e.g. `qwen3emb4b_batch1_...`), pass `--embeddings` explicitly to point at the matching `.pkl`. `scripts/run_all.sh` derives both paths from the same `EMBED_MODEL` + `EMBED_BATCH_SIZE` env vars so they stay in sync.

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

> ℹ️ **Why `--reasoning-parser qwen3` is in the script:** [scripts/bash.sh](scripts/bash.sh) ships with `--reasoning-parser qwen3` so that vLLM separates the model's `<think>...</think>` block into a dedicated `reasoning_content` field on each response. The judge stage reads from `reasoning_content`; without the parser flag everything ends up inside `content` and `STAGE1_REASONING` / `STAGE2_REASONING` come back empty. If you swap to a non-thinking model, you can drop the flag; if you swap to a different thinking model (e.g. DeepSeek-R1), change the parser name accordingly (and in vLLM < 0.9 add `--enable-reasoning`).

#### Step 2 — Run the judge stage

The bundled helper:

```bash
bash scripts/run_judge.sh
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
MAX_RETRIES=5 \
bash scripts/run_judge.sh
```

`scripts/run_judge.sh` exposes `BASE_URL`, `MODEL`, `API_KEY`, `TEMPERATURE`, `MAX_TOKENS`, `CONCURRENCY`, `BATCH_SIZE`, `CHECKPOINT_EVERY`, `MAX_RETRIES`, and `DATA_PATH`. Any extra CLI flag (e.g. `--agents human_lima_1 human_nyc_1`) can be appended after the script name and is forwarded to `python -m pipeline judge`.

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
| `--max-retries` | Retries per LLM call on transient failures (HTTP/JSON/timeout), with exponential backoff (2s → 4s → 8s) | `3` |
| `--agents` | Restrict judge to these agent names only (smoke-test mode). All outputs get a `_test` suffix so the full-run checkpoint isn't clobbered. | (none) |

##### Test mode with `--agents`

To validate end-to-end on a small subset without affecting the real checkpoint:

```bash
python -m pipeline judge --agents human_lima_1 human_nyc_1
```

This writes `llm_agreement_scores_test.parquet`, `comparable_pairs_test.csv`, etc. — fully isolated from the production artifacts.

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

- `llm_agreement_scores.parquet` — full per-pair scores (Stage 1 + Stage 2 + reasoning), the source of truth and resume checkpoint.
- `comparable_pairs.parquet` / `.csv` — pairs where `STAGE1_SCORE == 1`.
- `non_comparable_pairs.parquet` / `.csv` — pairs where `STAGE1_SCORE == 0` (genuinely non-comparable, not pending).
- `pending_pairs.parquet` / `.csv` — pairs where `STAGE1_SCORE` is NaN (still pending or failed after all retries). These are isolated so they don't pollute the non-comparable bucket; re-running the judge picks them up automatically via the resume logic.
- `summary.json` — totals, comparable ratio, agreement mean/std.
- `judge_scores.png` — heatmap grid (agent × agent, per block × sector). Symmetric — pairs are deduped at construction, then mirrored across the diagonal in the plot.

When using `--agents`, every file above gets a `_test` suffix.

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

# all-mpnet (light, CPU-friendly)
python scripts/generate_embeddings.py --model sentence-transformers/all-mpnet-base-v2 --resume

# Qwen3-Embedding-4B (large; bf16 + normalize recommended; trust_remote_code + left padding auto-set)
python scripts/generate_embeddings.py --model Qwen/Qwen3-Embedding-4B --device cuda --dtype bf16 --normalize --max-seq-length 512 --resume
```

The output filename is derived from `--model` and `--batch-size` (e.g. `qwen3emb4b_batch1_r2_embeddings_cache_keyed.pkl`). Pass `--output PATH` to override.

**Reproducibility:** the default `--batch-size 1` keeps embeddings byte-identical across runs. Larger batches can give different fp16/bf16 outputs on the same input due to non-deterministic accumulation order in CUDA matmul. Encode in batch only if reproducibility doesn't matter.

#### Relevant flags for `generate_embeddings.py`

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | HF model name | (required) |
| `--data` | Input CSV | `data/r2_cleaned.csv` |
| `--output` | Cache `.pkl` path | derived from model + batch |
| `--batch-size` | Encode batch size (1 = deterministic) | `1` |
| `--device` | `cpu` / `cuda` / `cuda:0` etc. | auto |
| `--normalize` | Normalize embeddings to unit norm | off |
| `--dtype` | `fp32` / `fp16` / `bf16` | model default |
| `--padding-side` | `left` / `right` (`left` required for Qwen3-Embedding) | auto |
| `--instruction` | Optional task instruction prefix (`Instruct: ...\nQuery: `) | (none) |
| `--max-seq-length` | Truncate to N tokens (overrides model default) | model default |
| `--trust-remote-code` | Allow custom model code (auto-enabled for Qwen3-Embedding) | off |
| `--resume` / `--no-resume` | Reuse existing cache as starting point | `--resume` |
| `--npz` | Also dump a numpy `.npz` alongside the pickle | (none) |

Optional `.npz` output for downstream tooling:

```bash
python scripts/generate_embeddings.py --model sentence-transformers/all-mpnet-base-v2 --npz external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.npz --resume
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
| `data/preprocess.log` | Per-row trace of every block-2 transformation: `AS-IS`, `EXTRACTED-XY` (matched `"X out of Y"`), `EXTRACTED-NUM` (picked last valid number), or `NAN` (couldn't parse). Use this when you want to audit a specific cleaning decision. |
| `data/preprocess_block2_audit.csv` | Tabular version of the same audit, one row per block-2 answer with columns `AGENT, VIDEO, QUESTION_NUM, REPETITION, ANSWER_RAW, ANSWER_CLEAN, ACTION`. Easier to grep/filter than the log when you want to look at *all* `nan_no_match` cases at once. |

The two CSVs share the exact same schema (`AGENT, VIDEO, BLOCK, QUESTION_NUM, REPETITION, ANSWER`).

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
│   ├── bias_violin_distribution.png         # original: per-agent color + legend
│   ├── bias_violin_distribution_numbered.png # uniform color + numbered x-ticks (use bias_agent_order.csv to decode)
│   └── bias_agent_order.csv                 # mapping NUMBER -> AGENT for the numbered variant
└── judge/
    ├── judge_scores.png
    ├── llm_agreement_scores.parquet  # cache (row-level resume)
    ├── comparable_pairs.parquet
    ├── comparable_pairs.csv
    ├── non_comparable_pairs.parquet
    ├── non_comparable_pairs.csv
    ├── pending_pairs.parquet         # rows with STAGE1_SCORE NaN (pending/failed)
    ├── pending_pairs.csv
    └── summary.json
```

---

## Packaging the cache for reuse

To bundle inputs + cached artifacts (preprocessed CSV, embedding caches, parquet checkpoints — *especially the judge checkpoint*) so you or someone else can replicate the pipeline locally without re-running the slow stages:

```bash
bash scripts/package_cache.sh                       # writes robusto_cache_<timestamp>.tar.gz at repo root
bash scripts/package_cache.sh /path/to/output.tar.gz
```

The archive includes `data/r2_cleaned.csv`, `data/raw/`, `final_questions_v3.yaml`, all top-level `external_embeds/*.pkl`, and every `outputs/pipeline/**/*.parquet`. It excludes plots (regenerable), `*/old/` dirs, `venv/`, `__pycache__/`, `.git/`.

To restore on another machine: extract at the repo root, then run any stage — the resume logic will skip everything already cached.

---

## Agent Ordering Convention

All heatmaps and violin plots use the same agent ordering, defined in [`pipeline/utils/metrics.py:get_ordered_agents`](pipeline/utils/metrics.py):

1. **VLMs**
2. **HUMAN_LIMA** (human annotators from Lima)
3. **HUMAN_NYC** (human annotators from NYC)

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