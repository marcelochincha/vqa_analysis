"""#11 -- LOCAL half of the judge run-to-run reliability check.

Two commands, both run here (no GPU):

  sample : draw a stratified subsample of pairs from the local judge parquet and
           write a small CSV (`judge_reliability_pairs.csv`). Upload ONLY that file
           to the GPU box.
             python judge_reliability_analyze.py sample --n-per-block 150

  report : after the GPU box returns `judge_reliability_runs_<model>.parquet`,
           compute per-block Krippendorff alpha and stability entirely locally.
             python judge_reliability_analyze.py report --runs-parquet <file>

This keeps all heavy data and all analysis on this machine; the remote only produces
a compact scores parquet.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)
DEFAULT_PARQUET = os.path.join(
    REPO, "outputs", "pipeline_non_embed", "judge", "llm_agreement_scores.parquet")
PAIRS_CSV = os.path.join(OUTDIR, "judge_reliability_pairs.csv")
BLOCK_NAME = {1: "Factual", 2: "Ratings", 3: "Counterfactual", 4: "Reasoning"}


# ------------------------------------------------------------------ sample
def cmd_sample(args):
    rng = np.random.default_rng(args.seed)
    d = pd.read_parquet(args.pairs_parquet)
    d = d[d.BLOCK.isin([1, 3, 4])].copy()          # judge does not score numeric B2
    keep = []
    for b, g in d.groupby("BLOCK"):
        take = min(args.n_per_block, len(g))
        keep.append(g.loc[rng.choice(g.index.to_numpy(), take, replace=False)])
    s = pd.concat(keep).reset_index(drop=True)
    cols = ["VIDEO", "QUESTION_NUM", "QUESTION", "VIDEO_SECTOR", "BLOCK",
            "AGENT_I", "ANSWER_I", "AGENT_J", "ANSWER_J", "PAIR_KEY"]
    s = s[[c for c in cols if c in s.columns]]
    s.to_csv(PAIRS_CSV, index=False)
    print(f"wrote {PAIRS_CSV}  ({len(s)} pairs, "
          f"blocks {s.BLOCK.value_counts().to_dict()})")
    print("-> upload this CSV to the GPU box and run judge_reliability_generate.py")


# ------------------------------------------------------------------ Krippendorff
def krippendorff_alpha(matrix, level="nominal"):
    """matrix: [coders(runs) x units(pairs)] with np.nan for missing values."""
    def delta(a, b):
        return (a - b) ** 2 if level == "interval" else (0.0 if a == b else 1.0)

    coinc = {}
    for unit in matrix.T:                      # iterate units (pairs)
        vals = unit[~np.isnan(unit)]
        m = len(vals)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for a, b in itertools.permutations(vals, 2):
            coinc[(a, b)] = coinc.get((a, b), 0.0) + w
    if not coinc:
        return np.nan
    classes = sorted({c for pair in coinc for c in pair})
    n_c = {c: 0.0 for c in classes}
    for (a, b), o in coinc.items():
        n_c[a] += o
    n = sum(n_c.values())
    if n <= 1:
        return np.nan
    Do = sum(coinc.get((a, b), 0.0) * delta(a, b) for a in classes for b in classes)
    De = sum(n_c[a] * n_c[b] * delta(a, b) for a in classes for b in classes) / (n - 1)
    return 1.0 - Do / De if De else np.nan


def _consistency(pv):
    cons = []
    for u in pv.T:
        v = u[~np.isnan(u)]
        cons.append(1.0 if len(v) and np.all(v == v[0]) else 0.0)
    return float(np.mean(cons)) if cons else np.nan


# ------------------------------------------------------------------ report
def cmd_report(args):
    runs = pd.read_parquet(args.runs_parquet)
    rows = []
    for b, g in runs.groupby("BLOCK"):
        piv1 = g.pivot(index="run", columns="pair", values="STAGE1").to_numpy(float)
        pivF = g.pivot(index="run", columns="pair", values="FINAL").to_numpy(float)
        finsd = np.nanmean([np.nanstd(u) for u in pivF.T if np.sum(~np.isnan(u)) > 1])
        rows.append(dict(
            block=int(b), block_name=BLOCK_NAME.get(int(b), str(b)),
            n_pairs=int(g.pair.nunique()), runs=int(g.run.nunique()),
            alpha_stage1_nominal=krippendorff_alpha(piv1, "nominal"),
            alpha_final_interval=krippendorff_alpha(pivF, "interval"),
            stage1_fully_consistent_frac=_consistency(piv1),
            final_fully_consistent_frac=_consistency(pivF),
            mean_final_sd_across_runs=float(finsd),
        ))
    summ = pd.DataFrame(rows)
    tag = os.path.splitext(os.path.basename(args.runs_parquet))[0].replace(
        "judge_reliability_runs_", "")
    summ.to_csv(os.path.join(OUTDIR, f"judge_reliability_{tag}.csv"), index=False)
    with open(os.path.join(OUTDIR, f"judge_reliability_{tag}.json"), "w") as f:
        json.dump(summ.to_dict(orient="records"), f, indent=2)
    print("=== run-to-run reliability ===")
    print(summ.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--pairs-parquet", default=DEFAULT_PARQUET)
    s.add_argument("--n-per-block", type=int, default=150)
    s.add_argument("--seed", type=int, default=20260810)
    s.set_defaults(func=cmd_sample)
    r = sub.add_parser("report")
    r.add_argument("--runs-parquet", required=True)
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
