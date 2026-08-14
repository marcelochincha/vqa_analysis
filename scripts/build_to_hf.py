"""Build to_hf/ upload bundle for Hugging Face.

Transforms applied to every output:
  * human_nyc_11..20 -> human_nyc_1..10 (so NYC mirrors Lima numbering).
  * AGENT* columns renamed to SYSTEM* (AGENT->SYSTEM, AGENT_I->SYSTEM_I, ...).

r2_cleaned is emitted as csv + parquet with column order
    VIDEO, BLOCK, QUESTION_NUM, SYSTEM, REPETITION, ANSWER
and rows sorted by VIDEO -> BLOCK -> QUESTION_NUM -> SYSTEM, where SYSTEM
follows the SAME ordering used in the plots (pipeline.utils.metrics.get_ordered_agents:
VLMs -> human_lima -> human_nyc, each natural-sorted).

Output layout (structure kept as left in to_hf/):
    to_hf/r2_cleaned.csv
    to_hf/r2_cleaned.parquet
    to_hf/embeds/<model>_embeddings.parquet
    to_hf/llm_judge_scores/<name>.parquet
    to_hf/human_validation/judge_human_sheet.csv   (hand-labeled judge-vs-human sample)
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.utils.metrics import _natural_key, get_ordered_agents  # noqa: E402

OUT = ROOT / "to_hf"
OUT.mkdir(exist_ok=True)

NYC_RE = re.compile(r"^human_nyc_(\d+)$")


def remap_agent(a):
    if not isinstance(a, str):
        return a
    m = NYC_RE.match(a)
    if m:
        return f"human_nyc_{int(m.group(1)) - 10}"
    return a


# --- 1. r2_cleaned -> csv + parquet (renamed, reordered, sorted) ---
df = pd.read_csv(ROOT / "data" / "r2_cleaned.csv")
df["AGENT"] = df["AGENT"].map(remap_agent)
df = df.rename(columns={"AGENT": "SYSTEM"})

# SYSTEM ordering identical to the plots.
system_order = get_ordered_agents(df["SYSTEM"].tolist())
video_order = sorted(df["VIDEO"].unique(), key=_natural_key)
df["SYSTEM"] = pd.Categorical(df["SYSTEM"], categories=system_order, ordered=True)
df["VIDEO"] = pd.Categorical(df["VIDEO"], categories=video_order, ordered=True)
df = df.sort_values(["VIDEO", "BLOCK", "QUESTION_NUM", "SYSTEM"]).reset_index(drop=True)

cols = ["VIDEO", "BLOCK", "QUESTION_NUM", "SYSTEM", "REPETITION", "ANSWER"]
df = df[cols]
# Write plain strings (not pandas categoricals) for portability.
df["VIDEO"] = df["VIDEO"].astype(str)
df["SYSTEM"] = df["SYSTEM"].astype(str)
df.to_csv(OUT / "r2_cleaned.csv", index=False)
df.to_parquet(OUT / "r2_cleaned.parquet", index=False)
print(f"[csv] rows={len(df)} systems={len(system_order)} videos={len(video_order)}")
print(f"      SYSTEM order: {system_order}")

# --- 2. embedding caches -> parquet ---
# key = (AGENT, VIDEO, QUESTION_NUM, REPETITION), value = float vector
embeds_out = OUT / "embeds"
embeds_out.mkdir(exist_ok=True)
embeds = {
    "allmpnet_batch1": "allmpnet_batch1_r2_embeddings_cache_keyed.pkl",
    "qwen3emb4b_batch1": "qwen3emb4b_batch1_r2_embeddings_cache_keyed.pkl",
}
for out_name, name in embeds.items():
    with open(ROOT / "external_embeds" / name, "rb") as f:
        d = pickle.load(f)
    rows = [
        (remap_agent(agent), video, int(qnum), int(rep), np.asarray(vec, dtype=np.float32))
        for (agent, video, qnum, rep), vec in d.items()
    ]
    edf = pd.DataFrame(
        rows, columns=["SYSTEM", "VIDEO", "QUESTION_NUM", "REPETITION", "embedding"]
    )
    edf.to_parquet(embeds_out / f"{out_name}_embeddings.parquet", index=False)
    print(f"[embed] {out_name} rows={len(edf)} dim={len(rows[0][4])}")

# --- 3. llm_judge parquet files ---
judge_dir = ROOT / "outputs" / "pipeline_non_embed" / "judge"
judge_out = OUT / "llm_judge_scores"
judge_out.mkdir(exist_ok=True)
for pq in sorted(judge_dir.glob("*.parquet")):
    jdf = pd.read_parquet(pq)
    for col in ("AGENT_I", "AGENT_J"):
        if col in jdf.columns:
            jdf[col] = jdf[col].map(remap_agent)
    if "PAIR_KEY" in jdf.columns:
        jdf["PAIR_KEY"] = jdf["PAIR_KEY"].map(
            lambda arr: [remap_agent(a) for a in arr] if arr is not None else arr
        )
    jdf = jdf.rename(columns={"AGENT_I": "SYSTEM_I", "AGENT_J": "SYSTEM_J"})
    jdf.to_parquet(judge_out / pq.name, index=False)
    print(f"[judge] {pq.name} rows={len(jdf)} cols_renamed=AGENT_I/J->SYSTEM_I/J")

# --- 4. human validation sheet (hand-labeled judge-vs-human sample) ---
# Primary data: cannot be regenerated. Backs the judge-human agreement (kappa)
# reported in the supplement. HUMAN_COMPARABLE/HUMAN_SCORE are the human labels;
# JUDGE_STAGE1/JUDGE_FINAL are the judge's own verdicts on the same pairs.
hv_src = ROOT / "rebuttal" / "outputs" / "judge_human_sheet.csv"
if hv_src.exists():
    hv_out = OUT / "human_validation"
    hv_out.mkdir(exist_ok=True)
    hv = pd.read_csv(hv_src)
    hv = hv[hv["HUMAN_COMPARABLE"].notna()].reset_index(drop=True)  # labeled only
    for col in ("AGENT_A", "AGENT_B"):
        if col in hv.columns:
            hv[col] = hv[col].map(remap_agent)
    hv = hv.rename(columns={"AGENT_A": "SYSTEM_A", "AGENT_B": "SYSTEM_B"})
    hv.to_csv(hv_out / "judge_human_sheet.csv", index=False)
    print(f"[human_val] rows={len(hv)} (labeled) -> human_validation/judge_human_sheet.csv")
else:
    print(f"[human_val] skipped: {hv_src} not found")

print("done ->", OUT)
