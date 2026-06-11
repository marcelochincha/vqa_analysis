"""Build the BLOC2BIN dataset + embeddings cache from allmpnet WITHOUT re-encoding.

Block 2 (Q6-Q10) answers are a 1-10 ordinal scale. This binarizes ALL of them
(>=6 -> "True", <6 -> "False", unparseable left as-is, e.g. "nan") and produces:

  1. data/r2_cleaned_block2_binary.csv  - the binarized CSV (only Block 2 changes).
  2. external_embeds/allmpnet_batch1_r2_block2bin_cache_keyed.pkl - a copy of the
     allmpnet cache where every Block-2 key is REMAPPED to the canonical embedding
     of its binary label ("True"/"False"/"nan").

No embedding model is run: the canonical True/False/nan vectors are looked up from
the existing q9q10binary cache (which already contains allmpnet embeddings of those
exact strings), so this is pure dict surgery.

Usage:
    python scripts/build_block2bin.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 6  # >= 6 -> True, < 6 -> False


def binarize(value: str) -> str:
    s = str(value).strip()
    try:
        n = int(float(s))
    except (ValueError, TypeError):
        return s  # leave 'nan' / unparseable untouched
    if 1 <= n <= 10:
        return "True" if n >= THRESHOLD else "False"
    return s


def extract_label_vectors(q9q10_csv: Path, q9q10_cache: dict) -> dict[str, np.ndarray]:
    """Pull the canonical embedding for each binary label from the q9q10binary cache."""
    binr = pd.read_csv(q9q10_csv, keep_default_na=False)
    by_label: dict[str, list] = {}
    for r in binr[binr["QUESTION_NUM"].isin([9, 10])].itertuples(index=False):
        k = (r.AGENT, r.VIDEO, r.QUESTION_NUM, r.REPETITION)
        if k in q9q10_cache:
            by_label.setdefault(r.ANSWER, []).append(q9q10_cache[k])
    vectors = {lab: np.mean(np.vstack(vs), axis=0) for lab, vs in by_label.items()}
    for lab, v in vectors.items():
        print(f"  canonical vector for {lab!r}: dim={v.shape[0]} norm={np.linalg.norm(v):.4f} (from {len(by_label[lab])} samples)")
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("data/r2_cleaned.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/r2_cleaned_block2_binary.csv"))
    parser.add_argument("--base-cache", type=Path, default=Path("external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl"))
    parser.add_argument("--label-cache", type=Path, default=Path("external_embeds/allmpnet_batch1_r2_q9q10binary_cache_keyed.pkl"))
    parser.add_argument("--label-csv", type=Path, default=Path("data/r2_cleaned_q9q10_binary.csv"))
    parser.add_argument("--output-cache", type=Path, default=Path("external_embeds/allmpnet_batch1_r2_block2bin_cache_keyed.pkl"))
    args = parser.parse_args()

    # --- 1. binarize all of Block 2 ---
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    mask = df["BLOCK"].astype(str) == "2"
    print(f"Loaded {len(df)} rows; Block-2 rows: {int(mask.sum())}")
    before = df.loc[mask, "ANSWER"].value_counts(dropna=False).to_dict()
    df.loc[mask, "ANSWER"] = df.loc[mask, "ANSWER"].apply(binarize)
    after = df.loc[mask, "ANSWER"].value_counts(dropna=False).to_dict()
    print(f"Block-2 before: {dict(sorted(before.items(), key=lambda x: str(x[0])))}")
    print(f"Block-2 after : {after}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved binarized CSV -> {args.output_csv}\n")

    # --- 2. build remapped cache (no recompute) ---
    print("Extracting canonical label vectors from q9q10binary cache:")
    with args.label_cache.open("rb") as f:
        label_cache = pickle.load(f)
    label_vecs = extract_label_vectors(args.label_csv, label_cache)

    with args.base_cache.open("rb") as f:
        cache = pickle.load(f)
    print(f"\nLoaded base allmpnet cache: {len(cache):,} entries")

    new_cache = dict(cache)
    remapped, fallback = 0, 0
    bin_lookup = {
        (r.AGENT, r.VIDEO, int(r.QUESTION_NUM), int(r.REPETITION)): r.ANSWER
        for r in df[mask].itertuples(index=False)
    }
    for key in cache:
        if key[2] in (6, 7, 8, 9, 10):  # Block-2 question numbers
            label = bin_lookup.get((key[0], key[1], int(key[2]), int(key[3])))
            if label in label_vecs:
                new_cache[key] = label_vecs[label]
                remapped += 1
            else:
                fallback += 1  # keep original embedding (unexpected label)

    print(f"Remapped {remapped:,} Block-2 entries to True/False/nan vectors; {fallback} left untouched")
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.output_cache.open("wb") as f:
        pickle.dump(new_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved BLOC2BIN cache -> {args.output_cache}")


if __name__ == "__main__":
    main()
