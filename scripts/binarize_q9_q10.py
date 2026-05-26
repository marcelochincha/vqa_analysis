"""Binarize Block-2 Q9 and Q10 answers (1-10 scale) into True/False.

Threshold: ANSWER >= 6 -> "True", <= 5 -> "False". Unparseable values ("nan")
are preserved as-is. All other rows are left untouched. Result is written to
a sibling CSV so r2_cleaned.csv stays intact.

Usage:
    python scripts/binarize_q9_q10.py
    python scripts/binarize_q9_q10.py --input data/r2_cleaned.csv --output data/r2_cleaned_q9q10_binary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_QUESTIONS = (9, 10)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/r2_cleaned.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/r2_cleaned_q9q10_binary.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    print(f"Loaded {len(df)} rows from {args.input}")

    mask = (df["BLOCK"].astype(str) == "2") & (df["QUESTION_NUM"].astype(str).isin([str(q) for q in TARGET_QUESTIONS]))
    target = df.loc[mask].copy()
    print(f"Target rows (Block 2, Q{TARGET_QUESTIONS[0]}/Q{TARGET_QUESTIONS[1]}): {len(target)}")

    before_counts = target["ANSWER"].value_counts(dropna=False).to_dict()
    df.loc[mask, "ANSWER"] = df.loc[mask, "ANSWER"].apply(binarize)
    after_counts = df.loc[mask, "ANSWER"].value_counts(dropna=False).to_dict()

    print("\nBefore (raw 1-10 distribution):")
    for k in sorted(before_counts, key=lambda x: (str(x).isdigit() is False, int(x) if str(x).isdigit() else 0)):
        print(f"  {k!r:>8}: {before_counts[k]}")

    print("\nAfter (binarized):")
    for k, v in sorted(after_counts.items()):
        print(f"  {k!r:>8}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
