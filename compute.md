# VQA Analysis Pipeline - Metric Computation Documentation

This document describes the computational pipeline for analyzing Visual Question Answering (VQA) data from human annotators and Vision-Language Models (VLMs). The pipeline consists of five sequential stages, each computing different metrics and generating visualizations.

---

## 1. Preprocess Stage

### Purpose
Prepares raw data from human annotators and VLM outputs into a unified CSV format suitable for downstream analysis. The stage also assigns experimental blocks and tracks repetition identifiers.

### Data Inputs
- **Human CSV**: Raw survey responses with Q&A columns in wide format (e.g., `R2_153-Q1`)
- **VLM JSON files**: Model-generated answers in JSON format

### Computation
1. **Human data transformation**:
   - Melts wide-format columns (`R2_153-Q1`) into long format (question-answer pairs)
   - Creates `ORIG_ROW` column to track original row positions
   - Extracts video identifiers from column names

2. **Repetition assignment**:
   - Each answer gets a unique `REPETITION` identifier (1-based)
   - Enables tracking of repeated measurements for same video-question pairs

3. **Block assignment**:
   - Questions are grouped into experimental blocks:
     - **Block 1**: Questions 1-5 (Factual - video identification)
     - **Block 2**: Questions 6-10 (Ratings - scale 1-10)
     - **Block 3**: Questions 11-15 (Counterfactual & Hypothetical)
     - **Block 4**: Questions 16+ (Reasoning)
   - Computed via `assign_block(question_num)` function

### Output
- `merged_data.csv`: Unified dataframe with columns:
  - `VIDEO`: Video identifier (e.g., `video_001`)
  - `AGENT`: Model or human identifier
  - `QUESTION_NUM`: Question number (1-10)
  - `ANSWER`: Numeric or text answer
  - `REPETITION`: Repetition number
  - `BLOCK`: Experimental block (1, 2, 3, or 4)
  - `VIDEO_SECTOR`: Geographic region (Lima or NYC)

---

## 2. Embed Stage (PCA Visualization)

### Purpose
Reduces high-dimensional embedding vectors to 2D for visualization using Principal Component Analysis (PCA). This reveals the spatial distribution of agent responses in an embedding space.

### Data Inputs
- Merged CSV from preprocess stage
- Embeddings cache (high-dimensional vectors per agent-video-question-repetition)

### Computation
1. **Embedding extraction**:
   - Loads cached embeddings for each (agent, video, question_num, repetition) tuple
   - Filters to repetition 1 for representative sampling

2. **PCA transformation**:
   - Fits PCA independently per block
   - Projects embeddings to 2D: `(pca_X, pca_Y)`
   - Block-specific PCA allows different dimensions per question type

3. **Agent grouping**:
   - Categorizes agents:
     - `vlm`: All non-human models
     - `lima`: Human annotators from Lima dataset
     - `nyc`: Human annotators from NYC dataset

### Visualization
- **Scatter plot grid**: Rows = sectors (Lima, NYC), Columns = blocks
- Points colored by agent group using predefined color palette
- Shared axis limits across all subplots for consistent comparison
- Legend showing all agent names
- Figure size: `6 * ncols x 5 * nrows`

### Output
- `pca_by_block_sector.png`: Grid of PCA scatter plots

---

## 3. Cosine Similarity Stage

### Purpose
Computes pairwise cosine similarity between agent embeddings for each video-question combination. This metric shows how similar the embedding representations are between different agents.

### Data Inputs
- Merged CSV from preprocess stage
- Embeddings cache

### Computation
1. **Grouping**:
   - Groups data by `(VIDEO, QUESTION_NUM)`
   - For each group, collects embeddings for all agents

2. **Cosine distance calculation**:
   - Uses `scipy.spatial.distance.cdist` with `metric="cosine"`
   - Converts to similarity: `sims = 1 - dists`

3. **Aggregation**:
   - Averages results across repetitions per `(VIDEO_SECTOR, AGENT_I, AGENT_J, BLOCK)`
   - Creates symmetric matrix for all agent pairs

4. **Agent ordering**:
   - Ordered groups: `VLMs → HUMAN_LIMA → HUMAN_NYC`
   - This places VLM agents first, human agents second

### Visualization
- **Heatmap grid**: Rows = sectors (Lima, NYC), Columns = blocks
- Colormap: Blue-red diverging palette via `sns.diverging_palette(220, 20)`
- Value range: `[-1, 1]` (full cosine similarity range)
- Square cells for symmetry
- Figure size: `6 * ncols x 5 * nrows`
- Title: "Cosine similarity heatmaps by block and region"

### Output
- `cosine_heatmap_grid.png`: Grid of cosine similarity heatmaps

---

## 4. RSA Stage (Representational Similarity Analysis)

### Purpose
Computes representational similarity between agent response patterns by comparing their representational geometric (RSA/RDM) matrices. Unlike cosine similarity (pairwise embedding comparison), RSA compares the *structure* of response spaces.

### Data Inputs
- Merged CSV from preprocess stage
- Embeddings cache

### Computation
1. **Gramian matrix computation**:
   - For each `(agent, block, sector)` combination:
   - Normalizes embeddings: `embeds_norm = embeds / ||embeds||`
   - Computes Gramian: `G = embeds_norm @ embeds_norm.T`
   - Represents pairwise inner products between responses

2. **Correlation calculation**:
   - For each `(agent_i, agent_j, block, sector)` pair:
   - Extracts upper triangle (k=1) from each Gramian
   - Computes Pearson correlation between the vectors
   - Correlation measures how similarly two agents structure their response patterns

3. **Agent ordering**: Same as cosine: `VLMs → HUMAN_LIMA → HUMAN_NYC`

### Interpretation
- High correlation → Agents have similar response geometry/spatial structure
- Low/correlation → Different response patterns
- Diagonal = 1.0 (self-correlation)

### Visualization
- **Heatmap grid**: Same structure as cosine
- Colormap: Blue-red diverging palette
- Value range: `[-1, 1]`
- Title: "RSA analysis - Representational Similarity Analysis"
- Figure size: `6 * ncols x 5 * nrows`

### Output
- `rsa_heatmap_grid.png`: Grid of RSA correlation heatmaps

---

## 5. Bias Stage (Human Consensus Comparison)

### Purpose
Analyzes VLM responses against human annotator consensus to detect systematic biases. Compares VLM answer distributions to human regional norms (Lima vs NYC).

### Data Inputs
- Merged CSV from preprocess stage (Block 2 data only: Questions 6-10)

### Computation
1. **Human consensus building**:
   - Filters to human agents only
   - Groups by `(QUESTION_NUM, HUMAN_REGION, VIDEO_REGION)`
   - Computes mean answer: `consensus = mean(human_answers)`
   - Creates separate baselines for Lima and NYC humans

2. **Statistical tests** (per VLM agent, question, sector):
   - **Wasserstein distance** (Earth Mover's Distance): Measures distribution shift between Lima and NYC video responses
   - **Kolmogorov-Smirnov test**: Non-parametric test for distribution differences
   - Reports: statistic, p-value

3. **Answer aggregation**:
   - Filters to Block 2, Repetition 1, VLM agents only
   - Groups by `(VIDEO_REGION, QUESTION_NUM, AGENT)`

### Visualization
- **FacetGrid violin plot**:
  - Rows: VIDEO_REGION (Lima, NYC)
  - Columns: QUESTION_NUM (6-10)
  - Hue: AGENT (VLM models)
  - Plot type: Violin plots with quartile inner

- **Consensus lines**:
  - Red dashed line: Lima human consensus
  - Blue dashed line: NYC human consensus
  - Reference lines showing regional human baselines

- **Question annotations** (below each column):
  - Full question text
  - Average Wasserstein distance between regions

- **Legend**:
  - Combined VLM agents + consensus lines
  - Title: "VLMs & Consensus"

- **FacetGrid parameters**:
  - `height=3, aspect=1.2`
  - Shared y-axis (1-10 rating scale)
  - Grid lines enabled

### Output
- `bias_violin_distribution.png`: Violin plots with consensus comparison

---

## Summary Table

| Stage | Metric | Computation | Visualization |
|-------|--------|-------------|----------------|
| Preprocess | Data transformation | Melt, block assignment, repetition tracking | N/A |
| Embed | PCA | Fit PCA per block, project to 2D | Scatter plot grid |
| Cosine | Cosine similarity | `1 - cdist(embeds, "cosine")` | Heatmap grid |
| RSA | Pearson correlation | `corr(Gi[upper], Gj[upper])` | Heatmap grid |
| Bias | Wasserstein + KS test | Distribution comparison vs. human consensus | Violin plot FacetGrid |

---

## Agent Ordering Convention

All heatmaps use consistent agent ordering for comparability:

1. **VLMs** (alphabetically sorted)
2. **HUMAN_LIMA** (alphabetically sorted)
3. **HUMAN_NYC** (alphabetically sorted)

This ordering places:
- First row/column: VLM agents
- Middle section: Lima human baseline
- Last section: NYC human baseline

---

## Technical Notes

### Color Schemes
- **Heatmaps**: `sns.diverging_palette(220, 20)` - Blue (low) to Red (high)
- **PCA**: Custom palette differentiating VLM, Lima human, NYC human
- **Violins**: Seaborn default "deep" palette for VLM agents

### Figure Sizing
- Grid layouts: `figsize=(6 * ncols, 5 * nrows)`
- Violin FacetGrid: `height=3, aspect=1.2`
- Output DPI: 150-300 depending on stage

### Data Filtering Conventions
- **RSA**: Repetition 1 only (representative sample)
- **Embed**: Repetition 1 only (representative sample)
- **Cosine**: All repetitions averaged
- **Bias**: Block 2, Repetition 1 (rating questions)