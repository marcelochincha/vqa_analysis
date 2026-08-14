"""Within-VLM repetition (run-to-run) consistency on the ALREADY-COLLECTED data.

No model is re-run. Each VLM answered every (video, question) 20 times; humans once.
The paper's analyses use repetition 1 only (verified in embed/cosine/rsa/bias/judge
stages). This module quantifies how much a VLM varies across its own 20 repetitions,
to (a) justify the single-repetition choice, (b) give a generation-reliability number
that complements the (GPU-bound) judge run-to-run check, and (c) test whether that
reliability itself depends on geography.

Two views:
  - Numeric (Block 2): SD across the 20 repetitions of each rating, vs the between-video
    signal SD -> a signal-to-noise argument that a single draw is representative.
  - Free-text (Blocks 1/3/4): mean pairwise cosine among the 20 repetition embeddings
    (self-consistency), vs the between-VLM similarity -> repetitions cluster far tighter
    than different systems do.

Outputs: rebuttal/outputs/repetition_consistency.{json,csv}
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data_io import (
    load_raw, load_ratings, load_embeddings, VIDEO_GEO, LIMA_VIDEOS, NYC_VIDEOS,
    system_type, is_vlm, CLOSED_VLMS, BLOCK_NAME, OUTDIR,
)

RNG = np.random.default_rng(20260813)
FREE_TEXT_BLOCKS = (1, 3, 4)


# ------------------------------------------------------------------ helpers
def _geo_perm_p(per_video: dict[str, float], n_perm: int = 10000):
    """Two-sided permutation p for a Lima-vs-NYC difference at the VIDEO level."""
    lima = np.array([per_video[v] for v in LIMA_VIDEOS if v in per_video])
    nyc = np.array([per_video[v] for v in NYC_VIDEOS if v in per_video])
    vals = np.concatenate([lima, nyc])
    n_l = len(lima)
    obs = float(abs(lima.mean() - nyc.mean()))
    count = 0
    for _ in range(n_perm):
        RNG.shuffle(vals)
        if abs(vals[:n_l].mean() - vals[n_l:].mean()) >= obs - 1e-12:
            count += 1
    return float(lima.mean()), float(nyc.mean()), obs, (count + 1) / (n_perm + 1)


def _mean_offdiag_cosine(vecs: np.ndarray) -> float:
    """Mean cosine over unordered pairs of the rows of `vecs` (already >=2 rows)."""
    n = vecs.astype(np.float32)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    n = n / norm
    sim = n @ n.T
    iu = np.triu_indices(len(n), k=1)
    return float(sim[iu].mean())


# ------------------------------------------------------------------ numeric
def numeric_consistency():
    d = load_ratings(rep=None)                      # all reps, Block 2, integer ratings
    d = d[d["system_type"] == "VLM"].copy()
    # SD across the 20 repetitions for each (agent, video, question)
    g = (d.groupby(["AGENT", "VIDEO", "QUESTION_NUM"])["rating"]
           .agg(rep_sd=lambda x: x.std(ddof=1), rep_mean="mean", n="count")
           .reset_index())
    g["video_geo"] = g["VIDEO"].map(VIDEO_GEO)
    g["closed"] = g["AGENT"].isin(CLOSED_VLMS)

    # between-video signal SD: SD of the per-(video,question) mean rating across videos
    per_vq = g.groupby(["VIDEO", "QUESTION_NUM"])["rep_mean"].mean().reset_index()
    signal_sd = float(per_vq.groupby("QUESTION_NUM")["rep_mean"].std(ddof=1).mean())

    res = {
        "mean_rep_sd": float(g["rep_sd"].mean()),
        "median_rep_sd": float(g["rep_sd"].median()),
        "between_video_signal_sd": signal_sd,
        "signal_to_noise": signal_sd / float(g["rep_sd"].mean()),
        "by_question": {int(q): round(float(v), 3)
                        for q, v in g.groupby("QUESTION_NUM")["rep_sd"].mean().items()},
        "by_vlm": {a: round(float(v), 3)
                   for a, v in g.groupby("AGENT")["rep_sd"].mean().sort_values().items()},
        "closed_vs_open_rep_sd": {
            "closed_T1.0": round(float(g[g.closed]["rep_sd"].mean()), 3),
            "open_T0.5": round(float(g[~g.closed]["rep_sd"].mean()), 3)},
    }
    # geography test on the per-video mean rep SD
    pv = g.groupby("VIDEO")["rep_sd"].mean().to_dict()
    lima_m, nyc_m, obs, p = _geo_perm_p(pv)
    res["geo_rep_sd"] = {"Lima": round(lima_m, 3), "NYC": round(nyc_m, 3),
                         "abs_diff": round(obs, 3), "perm_p": round(p, 3)}
    return res, g


# ------------------------------------------------------------------ free-text
def semantic_consistency(which="mpnet"):
    cache = load_embeddings(which)
    # index vectors by (agent, video, qnum) -> list of rep vectors, VLMs only, free-text
    from collections import defaultdict
    groups = defaultdict(list)
    rep1 = {}                                       # (video,qnum) -> {agent: rep1 vec}
    for (sysname, video, qnum, rep), vec in cache.items():
        if video not in VIDEO_GEO or not is_vlm(sysname):
            continue
        qn = int(qnum)
        block = 1 if qn <= 5 else 2 if qn <= 10 else 3 if qn <= 15 else 4
        if block not in FREE_TEXT_BLOCKS:
            continue
        groups[(sysname, video, qn, block)].append(np.asarray(vec, dtype=np.float32))
        if rep == 1:
            rep1.setdefault((video, qn), {})[sysname] = np.asarray(vec, dtype=np.float32)

    # within-VLM self-consistency: mean pairwise cosine among a system's 20 reps
    rows = []
    for (sysname, video, qn, block), vecs in groups.items():
        if len(vecs) < 2:
            continue
        rows.append(dict(agent=sysname, video=video, question_num=qn, block=block,
                         video_geo=VIDEO_GEO[video], closed=sysname in CLOSED_VLMS,
                         self_cos=_mean_offdiag_cosine(np.vstack(vecs))))
    sc = pd.DataFrame(rows)

    # between-VLM baseline: mean pairwise cosine across DIFFERENT systems (rep 1)
    betw = []
    for (video, qn), agentmap in rep1.items():
        qnb = 1 if qn <= 5 else 2 if qn <= 10 else 3 if qn <= 15 else 4
        if qnb not in FREE_TEXT_BLOCKS or len(agentmap) < 2:
            continue
        betw.append(dict(video=video, question_num=qn, block=qnb,
                         between_cos=_mean_offdiag_cosine(np.vstack(list(agentmap.values())))))
    bt = pd.DataFrame(betw)

    res = {
        "encoder": which,
        "mean_self_cosine": round(float(sc["self_cos"].mean()), 3),
        "mean_between_vlm_cosine": round(float(bt["between_cos"].mean()), 3),
        "by_block": {},
        "by_vlm": {a: round(float(v), 3)
                   for a, v in sc.groupby("agent")["self_cos"].mean().sort_values().items()},
        "closed_vs_open_self_cosine": {
            "closed_T1.0": round(float(sc[sc.closed]["self_cos"].mean()), 3),
            "open_T0.5": round(float(sc[~sc.closed]["self_cos"].mean()), 3)},
    }
    for b in FREE_TEXT_BLOCKS:
        s = sc[sc.block == b]["self_cos"].mean()
        w = bt[bt.block == b]["between_cos"].mean()
        res["by_block"][BLOCK_NAME[b]] = {
            "self": round(float(s), 3), "between_vlm": round(float(w), 3)}
    # geography test on per-video self-consistency
    pv = sc.groupby("video")["self_cos"].mean().to_dict()
    lima_m, nyc_m, obs, p = _geo_perm_p(pv)
    res["geo_self_cosine"] = {"Lima": round(lima_m, 3), "NYC": round(nyc_m, 3),
                              "abs_diff": round(obs, 3), "perm_p": round(p, 3)}
    return res, sc


# ------------------------------------------------------------------ main
def main():
    num, gnum = numeric_consistency()
    sem, gsem = semantic_consistency("mpnet")

    out = {"numeric_block2": num, "semantic_freetext": sem}
    with open(os.path.join(OUTDIR, "repetition_consistency.json"), "w") as f:
        json.dump(out, f, indent=2)
    gnum.to_csv(os.path.join(OUTDIR, "repetition_consistency_numeric.csv"), index=False)
    gsem.to_csv(os.path.join(OUTDIR, "repetition_consistency_semantic.csv"), index=False)

    print("=== Numeric (Block 2) within-VLM repetition stability ===")
    print(f"  mean within-VLM rep SD      = {num['mean_rep_sd']:.3f} rating points")
    print(f"  between-video signal SD     = {num['between_video_signal_sd']:.3f}")
    print(f"  signal-to-noise             = {num['signal_to_noise']:.2f}x")
    print(f"  closed T1.0 vs open T0.5    = {num['closed_vs_open_rep_sd']}")
    print(f"  geography (rep SD)          = {num['geo_rep_sd']}")
    print("\n=== Free-text (Blocks 1/3/4) semantic self-consistency [mpnet] ===")
    print(f"  within-VLM self cosine      = {sem['mean_self_cosine']:.3f}")
    print(f"  between-VLM cosine (rep1)   = {sem['mean_between_vlm_cosine']:.3f}")
    print(f"  by block                    = {sem['by_block']}")
    print(f"  closed T1.0 vs open T0.5    = {sem['closed_vs_open_self_cosine']}")
    print(f"  geography (self cosine)     = {sem['geo_self_cosine']}")
    print("\nwrote repetition_consistency.{json,csv}")


if __name__ == "__main__":
    main()
