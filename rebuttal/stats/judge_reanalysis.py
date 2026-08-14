"""#8 / #10 / #11 / #14 -- Re-analysis of the LLM-as-a-judge scores (no re-run).

All four are post-hoc over the existing per-pair parquet
(outputs/pipeline_non_embed/judge/llm_agreement_scores.parquet):

  #8  Reconcile the pair counts. The paper reports 63,121 / 57,886 / 61,302
      comparable pairs (B1/B3/B4); the current deduplicated run has a fixed
      43,500 pairs/block = C(30,2)*100 and different comparable counts. We print
      the correct numbers and the comparability rate pi^(b,r).
  #10 Family ablation. Recompute the agreement excluding pairs that touch
      Qwen3-VL-8B (and, in a second pass, Cosmos-Reason2-8B) -- both share a family
      with the Qwen3-4B judge and the Qwen3-Embedding-4B encoder -- and check
      whether the judge scores those systems more generously.
  #11 Inference-budget split. Closed Gemini (1 FPS, T=1.0) vs open models
      (10 FPS, T=0.5): agreement within each group and Human-vs-group.
  #14 Rubric rescale. The Table-4 scale has no 0 and counts -1 ("agree on the
      object/action, differ on degree") as disagreement. Re-map STAGE2 under three
      alternative scales and report how much the cell means move.

Also included: a permutation test of whether mean judge agreement differs by video
geography (ties the judge modality into the #1 geography question).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data_io import (OUTDIR, load_judge, system_type, CLOSED_VLMS, QWEN_FAMILY,
                     is_vlm, BLOCK_NAME)

RNG = np.random.default_rng(20260810)
N_PERM = 10_000

# paper-reported comparable counts, for the #8 reconciliation table
PAPER_COMPARABLE = {1: 63121, 3: 57886, 4: 61302}

RESCALES = {
    "paper_identity": {2: 2, 1: 1, -1: -1, -2: -2, 0: 0},
    "symmetric_soft": {2: 1.0, 1: 0.5, -1: -0.5, -2: -1.0, 0: 0.0},
    "minus1_partial_agree": {2: 1.0, 1: 0.75, -1: 0.25, -2: 0.0, 0: 0.0},
    "minus1_neutral": {2: 2, 1: 1, -1: 0, -2: -2, 0: 0},      # isolates the -1 recode
    "minus1_as_agree": {2: 2, 1: 1, -1: 1, -2: -2, 0: 0},     # -1 counted as agreement
}


def _annotate(d):
    d = d.copy()
    d["st_i"] = d["AGENT_I"].map(system_type)
    d["st_j"] = d["AGENT_J"].map(system_type)

    def ptype(r):
        a, b = sorted([r.st_i, r.st_j])
        if a.startswith("Human") and b.startswith("Human"):
            return "Human-Human"
        if a == "VLM" and b == "VLM":
            return "VLM-VLM"
        return "Human-VLM"
    d["pair_type"] = d.apply(ptype, axis=1)
    d["touches_qwenvl"] = (d.AGENT_I == "Qwen3-VL-8B-Instruct") | (d.AGENT_J == "Qwen3-VL-8B-Instruct")
    d["touches_qwenfam"] = d.AGENT_I.isin(QWEN_FAMILY) | d.AGENT_J.isin(QWEN_FAMILY)
    return d


def comparable(d):
    return d[d.STAGE1_SCORE == 1].copy()


# ------------------------------------------------------------------ #8
def reconcile_counts(d):
    rows = []
    for b in sorted(d.BLOCK.unique()):
        db = d[d.BLOCK == b]
        total = len(db)
        comp = int((db.STAGE1_SCORE == 1).sum())
        rows.append(dict(
            block=b, block_name=BLOCK_NAME[b], total_pairs=total,
            comparable=comp, comparability_rate=round(comp / total, 4),
            paper_reported_comparable=PAPER_COMPARABLE.get(b),
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #10 / #11
def agreement_by(d, groupcols):
    c = comparable(d)
    g = c.groupby(groupcols)["STAGE2_SCORE"].agg(["mean", "count"]).reset_index()
    return g.rename(columns={"mean": "J", "count": "n_pairs"})


def family_ablation(d):
    c = comparable(d)
    out = {}
    out["all"] = float(c.STAGE2_SCORE.mean())
    out["excl_qwenvl"] = float(c[~c.touches_qwenvl].STAGE2_SCORE.mean())
    out["excl_qwenfamily"] = float(c[~c.touches_qwenfam].STAGE2_SCORE.mean())
    # judge favouritism: agreement on VLM-VLM pairs that include a Qwen-family model
    vv = c[c.pair_type == "VLM-VLM"]
    out["VLMVLM_with_qwenfam"] = float(vv[vv.touches_qwenfam].STAGE2_SCORE.mean())
    out["VLMVLM_without_qwenfam"] = float(vv[~vv.touches_qwenfam].STAGE2_SCORE.mean())
    # Human-VLM agreement: is the Qwen-family VLM judged closer to humans?
    hv = c[c.pair_type == "Human-VLM"]
    out["HumanVLM_with_qwenfam"] = float(hv[hv.touches_qwenfam].STAGE2_SCORE.mean())
    out["HumanVLM_without_qwenfam"] = float(hv[~hv.touches_qwenfam].STAGE2_SCORE.mean())
    # per-block version of the ablation
    per_block = {}
    for b in sorted(c.BLOCK.unique()):
        cb = c[c.BLOCK == b]
        per_block[BLOCK_NAME[b]] = dict(
            all=float(cb.STAGE2_SCORE.mean()),
            excl_qwenfamily=float(cb[~cb.touches_qwenfam].STAGE2_SCORE.mean()),
        )
    out["per_block"] = per_block
    return out


def budget_split(d):
    c = comparable(d)
    c = c.copy()
    def budget(agent):
        if not is_vlm(agent):
            return "Human"
        return "closed_1fps" if agent in CLOSED_VLMS else "open_10fps"
    c["bud_i"] = c.AGENT_I.map(budget); c["bud_j"] = c.AGENT_J.map(budget)
    out = {}
    # within-group VLM agreement
    for grp, members in [("closed_1fps", CLOSED_VLMS)]:
        sub = c[c.AGENT_I.isin(members) & c.AGENT_J.isin(members)]
        out[f"{grp}_within"] = float(sub.STAGE2_SCORE.mean()) if len(sub) else None
    openv = c[(c.bud_i == "open_10fps") & (c.bud_j == "open_10fps")]
    out["open_10fps_within"] = float(openv.STAGE2_SCORE.mean())
    # Human vs each budget group
    for grp in ["closed_1fps", "open_10fps"]:
        m = (((c.bud_i == "Human") & (c.bud_j == grp)) |
             ((c.bud_j == "Human") & (c.bud_i == grp)))
        out[f"Human_vs_{grp}"] = float(c[m].STAGE2_SCORE.mean())
    return out


# ------------------------------------------------------------------ #14
def rubric_rescale(d):
    c = comparable(d)
    rows = []
    for name, mapping in RESCALES.items():
        s = c["STAGE2_SCORE"].map(mapping)
        row = {"rescale": name, "overall_J": float(s.mean())}
        for b in sorted(c.BLOCK.unique()):
            row[BLOCK_NAME[b]] = float(s[c.BLOCK == b].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ geo test
def geo_permutation(d, n_perm=N_PERM):
    """Does mean judge agreement differ Lima vs NYC? Permute video->geo labels."""
    c = comparable(d)
    results = {}
    for b in sorted(c.BLOCK.unique()):
        cb = c[c.BLOCK == b]
        vids = cb.VIDEO.to_numpy()
        uniq = np.array(sorted(cb.VIDEO.unique()))
        sect = cb.drop_duplicates("VIDEO").set_index("VIDEO")["VIDEO_SECTOR"]
        n_lima = int((sect == "Lima").sum())
        s = cb.STAGE2_SCORE.to_numpy()
        obs = abs(s[cb.VIDEO_SECTOR.values == "Lima"].mean()
                  - s[cb.VIDEO_SECTOR.values == "NYC"].mean())
        ge = 0
        for _ in range(n_perm):
            lima = set(RNG.choice(uniq, n_lima, replace=False).tolist())
            mask = np.isin(vids, list(lima))
            diff = abs(s[mask].mean() - s[~mask].mean())
            ge += (diff >= obs)
        results[BLOCK_NAME[b]] = dict(obs_gap=float(obs), p_perm=float((1 + ge) / (1 + n_perm)))
    return results


def main(n_perm=N_PERM):
    d = _annotate(load_judge())

    counts = reconcile_counts(d)
    counts.to_csv(os.path.join(OUTDIR, "judge_counts.csv"), index=False)

    by_type = agreement_by(d, ["BLOCK", "VIDEO_SECTOR", "pair_type"])
    by_type["block_name"] = by_type.BLOCK.map(BLOCK_NAME)
    by_type.to_csv(os.path.join(OUTDIR, "judge_agreement_by_type.csv"), index=False)

    fam = family_ablation(d)
    bud = budget_split(d)
    resc = rubric_rescale(d)
    resc.to_csv(os.path.join(OUTDIR, "judge_rubric_rescale.csv"), index=False)
    geo = geo_permutation(d, n_perm)

    report = {"family_ablation_10": fam, "budget_split_11": bud,
              "geo_permutation": geo}
    with open(os.path.join(OUTDIR, "judge_reanalysis.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=== #8 counts ===")
    print(counts.to_string(index=False))
    print("\n=== #10 family ablation ===")
    print(json.dumps({k: v for k, v in fam.items() if k != "per_block"}, indent=2))
    print("\n=== #11 budget split ===")
    print(json.dumps(bud, indent=2))
    print("\n=== #14 rubric rescale ===")
    print(resc.round(4).to_string(index=False))
    print("\n=== geo permutation on judge agreement ===")
    print(json.dumps(geo, indent=2))


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PERM
    main(n)
