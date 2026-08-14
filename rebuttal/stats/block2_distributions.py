"""#3 / #12 -- Distributional inference on Block-2 ratings.

#3  The paper reports the Wasserstein-1 distance and the KS statistic (Eq. 8/9)
    for each VLM's Lima-vs-NYC rating distributions but never the p-values or CIs
    the reviewer asks for. Here, per (VLM, question):
      * KS two-sample test  -> D_KS + exact p  (H0: same Lima/NYC distribution),
        Benjamini-Hochberg corrected across the VLM x question grid.
      * Wasserstein-1        -> 95% CI by video-cluster bootstrap
                              -> p by label-permutation null (shuffle Lima/NYC).
    VLMs contribute all 20 repetitions per video (richer CDF); the answer is the
    integer rating already extracted in preprocess.

#12 The bias figure compares the mean of 10 humans against VLM distributions that
    still carry the VLM's own sampling noise (20 reps). We give a symmetric,
    paired view: collapse each VLM to a per-video central tendency (median over
    reps) so both sides are one value per video, and bootstrap the human consensus
    and the VLM reps jointly. We report the naive vs matched Human-VLM L1 gap so the
    reader sees the asymmetry does not drive the conclusion.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from data_io import OUTDIR, load_ratings, is_vlm

RNG = np.random.default_rng(20260810)
N_BOOT = 10_000
N_PERM = 10_000
QUESTIONS = [6, 7, 8, 9, 10]


def _bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[idx] * n / k)
        adj[idx] = prev
    return adj


# ------------------------------------------------------------------ #3
def w1_bootstrap_ci(lima_by_vid, nyc_by_vid, n_boot=N_BOOT):
    lv = list(lima_by_vid); nv = list(nyc_by_vid)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        L = np.concatenate([lima_by_vid[v] for v in RNG.choice(lv, len(lv), replace=True)])
        N = np.concatenate([nyc_by_vid[v] for v in RNG.choice(nv, len(nv), replace=True)])
        boots[b] = stats.wasserstein_distance(L, N)
    return tuple(np.percentile(boots, [2.5, 97.5]))


def w1_perm_p(lima_vals, nyc_vals, obs, n_perm=N_PERM):
    pool = np.concatenate([lima_vals, nyc_vals])
    n_l = len(lima_vals)
    ge = 0
    for _ in range(n_perm):
        RNG.shuffle(pool)
        w = stats.wasserstein_distance(pool[:n_l], pool[n_l:])
        ge += (w >= obs)
    return (1 + ge) / (1 + n_perm)


def ks_wasserstein():
    r = load_ratings(rep=None)          # all reps
    vlms = sorted([a for a in r.AGENT.unique() if is_vlm(a)])
    rows = []
    for vlm in vlms:
        d = r[r.AGENT == vlm]
        for q in QUESTIONS:
            dq = d[d.QUESTION_NUM == q]
            lima = dq[dq.video_geo == "Lima"]
            nyc = dq[dq.video_geo == "NYC"]
            lv = lima["rating"].to_numpy(); nv = nyc["rating"].to_numpy()
            if len(lv) < 2 or len(nv) < 2:
                continue
            ks = stats.ks_2samp(lv, nv)
            w1 = stats.wasserstein_distance(lv, nv)
            lima_by = {v: g["rating"].to_numpy() for v, g in lima.groupby("VIDEO")}
            nyc_by = {v: g["rating"].to_numpy() for v, g in nyc.groupby("VIDEO")}
            ci = w1_bootstrap_ci(lima_by, nyc_by)
            pw = w1_perm_p(lv, nv, w1)
            rows.append(dict(
                VLM=vlm, question=f"Q{q}",
                n_lima=len(lv), n_nyc=len(nv),
                KS_D=float(ks.statistic), KS_p=float(ks.pvalue),
                W1=float(w1), W1_lo=float(ci[0]), W1_hi=float(ci[1]), W1_perm_p=float(pw),
            ))
    df = pd.DataFrame(rows)
    df["KS_p_BH"] = _bh(df["KS_p"].values)
    df["W1_perm_p_BH"] = _bh(df["W1_perm_p"].values)
    return df


# ------------------------------------------------------------------ #12
def asymmetry_paired():
    """Naive vs matched Human-VLM L1 gap per (question, region).

    naive   : |mean_humans(region) - vlm_rating| over ALL 20 vlm reps & videos.
    matched : collapse each vlm to a per-video median over reps, so both sides are
              one value per video; bootstrap over humans, reps and videos jointly.
    """
    r_all = load_ratings(rep=None)
    r1 = load_ratings(rep=1)
    hum = r1[r1.system_type.str.startswith("Human")]
    vlms = sorted([a for a in r_all.AGENT.unique() if is_vlm(a)])
    rows = []
    for region in ["Lima", "NYC"]:
        for q in QUESTIONS:
            h = hum[(hum.video_geo == region) & (hum.QUESTION_NUM == q)]
            # human consensus per video
            hcons = h.groupby("VIDEO")["rating"].mean()
            v_all = r_all[(r_all.video_geo == region) & (r_all.QUESTION_NUM == q) &
                          (r_all.AGENT.isin(vlms))]
            # ---- naive gap (all reps vs the fixed human consensus mean)
            naive_vals = []
            for _, row in v_all.iterrows():
                if row.VIDEO in hcons.index:
                    naive_vals.append(abs(hcons[row.VIDEO] - row.rating))
            naive = float(np.mean(naive_vals)) if naive_vals else np.nan
            # ---- matched gap (per-video vlm median), bootstrap CI over videos
            vmed = v_all.groupby(["AGENT", "VIDEO"])["rating"].median().reset_index()
            merged = vmed.merge(hcons.rename("hcons"), left_on="VIDEO", right_index=True)
            merged["gap"] = (merged["hcons"] - merged["rating"]).abs()
            matched = float(merged["gap"].mean())
            vids = merged["VIDEO"].unique()
            boots = np.empty(2000)
            for b in range(2000):
                bv = RNG.choice(vids, len(vids), replace=True)
                boots[b] = merged[merged.VIDEO.isin(bv)].groupby("VIDEO")["gap"].mean().mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(region=region, question=f"Q{q}",
                             naive_L1=naive, matched_L1=matched,
                             matched_lo=float(lo), matched_hi=float(hi)))
    return pd.DataFrame(rows)


def main():
    ksw = ks_wasserstein()
    ksw.to_csv(os.path.join(OUTDIR, "block2_ks_wasserstein.csv"), index=False)
    asym = asymmetry_paired()
    asym.to_csv(os.path.join(OUTDIR, "block2_asymmetry.csv"), index=False)

    summary = {
        "n_cells": int(len(ksw)),
        "KS_significant_raw_(p<.05)": int((ksw.KS_p < 0.05).sum()),
        "KS_significant_BH_(p<.05)": int((ksw.KS_p_BH < 0.05).sum()),
        "W1_perm_significant_BH": int((ksw.W1_perm_p_BH < 0.05).sum()),
        "W1_median": float(ksw.W1.median()),
        "per_question_mean_W1": ksw.groupby("question")["W1"].mean().round(4).to_dict(),
    }
    with open(os.path.join(OUTDIR, "block2_distributions_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("wrote block2_ks_wasserstein.csv / block2_asymmetry.csv")
    print(json.dumps(summary, indent=2))
    print("\nKS cells with BH p<.05:")
    sig = ksw[ksw.KS_p_BH < 0.05][["VLM", "question", "KS_D", "KS_p", "KS_p_BH", "W1"]]
    print(sig.to_string(index=False) if len(sig) else "  (none)")
    print("\n#12 asymmetry (naive vs matched L1), first rows:")
    print(asym.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
