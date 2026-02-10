# VQA Analysis: Preprocessing & Metrics

This project analyzes and compares answers from Vision-Language Models (VLMs) and humans on VQA tasks. It includes scripts for preprocessing data and computing various metrics and analyses.

## 1. Preprocessing

Preprocessing scripts are in the `pre_processing/` folder. They combine and clean human and VLM answers into a single CSV for analysis.

### Steps:

1. **Prepare Human Answers**
   - Edit or use the script: `pre_processing/pre_process_human.py`
   - Input: Raw human answers CSV (e.g., `pre_processing/humans/answers_raw_human.csv`)
   - Output: Cleaned CSV (e.g., `pre_processing/answers_human.csv`)

2. **Prepare VLM Answers**
   - Edit or use the script: `pre_processing/pre_process_vlms.py`
   - Input: VLM JSON files (e.g., `pre_processing/vlms/*.json`)
   - Output: Cleaned CSV (e.g., `pre_processing/answers_vlms.csv`)

3. **Combine All Answers**
   - Merge human and VLM CSVs into a single file: `pre_processing/answers_allagents.csv`
   - Columns: `AGENT, VIDEO, QUESTION_NUM, QUESTION, ANSWER`

See `pre_processing/readme.md` for more details.

## 2. Metric Computation & Analysis

Analysis scripts are in the `src/` folder. Each script computes a different metric or visualization.

### Main Scripts:

- `src/embed_analysis.py` — Embedding analysis (UMAP/PCA)
- `src/bias_analysis.py` — Bias analysis by region/agent
- `src/heatmap_smatch.py` — SMATCH metric (AMR parsing)
- `src/heatmap_stsb.py` — STSB-RoBERTa metric
- `src/run_all.py` — Run all analyses in sequence

### Usage

Run scripts from the workspace root:

```powershell
# Embedding analysis
python src/embed_analysis.py

# Bias analysis
python src/bias_analysis.py

# SMATCH metric
python src/heatmap_smatch.py

# STSB-RoBERTa metric
python src/heatmap_stsb.py

# Run all analyses
python src/run_all.py
```

Outputs are saved in the `outputs/` folder, organized by metric.

## 3. Requirements

Install dependencies:

```powershell
pip install -r requirements.txt
```

For SMATCH/AMR parsing:

```powershell
python -m amrlib.download model_parse_xfm_bart_large
```

---
Edit `src/config.py` to adjust paths, agent groups, or plotting parameters as needed.# VQA analysis