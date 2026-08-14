"""#6 -- Response length / style analysis (numbers only, no VLM re-run).

Blocks 1, 3 and 4 forced the VLMs to "limit your response to a single sentence"
while humans answered freely on Google Forms. Cosine, RSA and the LLM judge are all
sensitive to length and register, so a Human-vs-VLM gap could be partly a length
artifact. We quantify that with existing data:

  Part 1 -- length distributions per (system_type, block, region): word-count
            summaries + a Kruskal-Wallis test that the three system types differ,
            with the epsilon^2 effect size.
  Part 2 -- length as a confound on the judge: per comparable pair, does the absolute
            word-count difference between the two answers predict the agreement score
            (Spearman) and the comparability decision? Reported overall and for
            Human-VLM pairs specifically.

We do NOT re-run any block without the length restriction (that would need model
inference); that part is argued in the rebuttal. Block 2 is numeric and excluded.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from data_io import OUTDIR, load_raw, load_judge, system_type, BLOCK_NAME

FREE_TEXT_BLOCKS = [1, 3, 4]


def _wc(s):
    return len(str(s).split())


# ------------------------------------------------------------------ Part 1
def length_distributions():
    d = load_raw()
    d = d[d.BLOCK.isin(FREE_TEXT_BLOCKS)].copy()
    d = d[d.REPETITION == 1]
    d["wc"] = d["ANSWER"].map(_wc)
    rows = []
    for (block, region), g in d.groupby(["BLOCK", "video_geo"]):
        for st, gg in g.groupby("system_type"):
            wc = gg["wc"].to_numpy()
            rows.append(dict(
                block=block, block_name=BLOCK_NAME[block], region=region,
                system_type=st, n=len(wc),
                mean_wc=float(wc.mean()), median_wc=float(np.median(wc)),
                sd_wc=float(wc.std()), p10=float(np.percentile(wc, 10)),
                p90=float(np.percentile(wc, 90)),
            ))
    dist = pd.DataFrame(rows)

    # Kruskal-Wallis across the 3 system types, per block (pooled regions)
    kw = []
    for block, g in d.groupby("BLOCK"):
        groups = [gg["wc"].to_numpy() for _, gg in g.groupby("system_type")]
        H, p = stats.kruskal(*groups)
        N = len(g)
        eps2 = (H - len(groups) + 1) / (N - len(groups))   # epsilon-squared
        kw.append(dict(block=block, block_name=BLOCK_NAME[block],
                       H=float(H), p=float(p), epsilon2=float(eps2), N=int(N),
                       mean_wc_by_type={k: round(float(v), 1) for k, v in
                                        g.groupby("system_type")["wc"].mean().items()}))
    return dist, kw


# ------------------------------------------------------------------ Part 2
def length_confound_judge():
    j = load_judge()
    j = j.copy()
    j["wc_i"] = j["ANSWER_I"].map(_wc)
    j["wc_j"] = j["ANSWER_J"].map(_wc)
    j["wc_absdiff"] = (j["wc_i"] - j["wc_j"]).abs()
    j["st_i"] = j["AGENT_I"].map(system_type)
    j["st_j"] = j["AGENT_J"].map(system_type)
    j["is_human_vlm"] = ((j.st_i == "VLM") ^ (j.st_j == "VLM")) & \
                        ~((j.st_i == "VLM") & (j.st_j == "VLM"))

    out = {}
    comp = j[j.STAGE1_SCORE == 1]
    # Does length difference predict the agreement score?
    for label, sub in [("all_comparable", comp),
                       ("human_vlm_comparable", comp[comp.is_human_vlm])]:
        rho, p = stats.spearmanr(sub["wc_absdiff"], sub["STAGE2_SCORE"])
        out[label] = dict(n=int(len(sub)), spearman_rho=float(rho), p=float(p))
    # Does length difference predict whether a pair is judged comparable at all?
    rho_c, p_c = stats.spearmanr(j["wc_absdiff"], j["STAGE1_SCORE"])
    out["comparability_vs_lengthdiff"] = dict(
        n=int(len(j)), spearman_rho=float(rho_c), p=float(p_c))
    # mean length gap by pair type (context)
    j["pair_hv"] = np.where(j.is_human_vlm, "Human-VLM", "same-family")
    out["mean_wc_absdiff_by_pairkind"] = {
        k: round(float(v), 2) for k, v in j.groupby("pair_hv")["wc_absdiff"].mean().items()}
    return out


def main():
    dist, kw = length_distributions()
    dist.to_csv(os.path.join(OUTDIR, "length_distributions.csv"), index=False)
    conf = length_confound_judge()
    report = {"kruskal_by_block": kw, "judge_length_confound": conf}
    with open(os.path.join(OUTDIR, "length_style.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=== #6 length distributions (word count, rep1, free-text blocks) ===")
    print(dist.round(1).to_string(index=False))
    print("\n=== Kruskal-Wallis (system types differ in length?) ===")
    for k in kw:
        print(f"  {k['block_name']:15s} H={k['H']:.1f} p={k['p']:.2e} "
              f"eps2={k['epsilon2']:.3f}  means={k['mean_wc_by_type']}")
    print("\n=== length as a confound on the judge ===")
    print(json.dumps(conf, indent=2))


if __name__ == "__main__":
    main()
