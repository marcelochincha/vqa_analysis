"""#4 -- Noise ceilings and bootstrap CIs for the RSA comparisons (Fig. 3B).

RSA (Fig. 3B) currently shows raw system-vs-system correlations with no notion of
how much of the structure is signal vs measurement noise. This module adds the two
things the reviewer asked for:

  * A subject-level NOISE CEILING (Nili et al. 2014) built from the human RDMs:
      upper = mean over humans of corr(RDM_h, mean-human-RDM incl. h)
      lower = mean over humans of corr(RDM_h, mean-human-RDM excl. h)  [leave-one-out]
    The band [lower, upper] is the best any model could do given human disagreement.
  * BOOTSTRAP CIs over items (stimuli) for the mean VLM-to-human-consensus RSA and
    the human-human / VLM-VLM RSA, so each Fig-3B-style number gets an interval.

Per system, the RDM is the item x item cosine-similarity matrix over that system's
answer embeddings (repetition 1), matching the paper's Eq. (5). RSA between two RDMs
is the Pearson correlation of their upper-triangular entries (Eq. 6).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data_io import OUTDIR, embeddings_frame, BLOCK_NAME, system_type

RNG = np.random.default_rng(20260810)
N_BOOT = 5_000


def _rdm_vectors(ef_block_region):
    """Return dict system -> (item_matrix [n_items x d], ordered item keys)."""
    systems = {}
    item_keys = None
    for sysname, g in ef_block_region.groupby("system"):
        g = g.sort_values(["video", "question_num"])
        keys = list(zip(g["video"], g["question_num"]))
        M = np.vstack(g["vec"].to_numpy())
        systems[sysname] = (M, keys)
        if item_keys is None:
            item_keys = keys
    # keep only systems that cover the full common item set
    common = set(item_keys)
    systems = {s: v for s, v in systems.items() if set(v[1]) == common}
    return systems, item_keys


def _full_sim(M):
    """Full item x item cosine-similarity matrix for one system."""
    Xn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def _upper(S, idx, ii, jj):
    """Upper-tri of the resampled RDM S[idx][:,idx] at positions (ii,jj)."""
    return S[idx[ii], idx[jj]]


def _valid_pairs(idx):
    """Upper-tri index pairs (k=1) excluding duplicated items (idx[a]==idx[b]).

    Resampling items with replacement would otherwise inject cosine=1 off-diagonal
    entries and inflate every RSA correlation; masking those pairs removes the bias.
    """
    n = len(idx)
    ii, jj = np.triu_indices(n, k=1)
    keep = idx[ii] != idx[jj]
    return ii[keep], jj[keep]


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) / d) if d > 0 else np.nan


def block_region_stats(block, region, which="mpnet", n_boot=N_BOOT):
    ef = embeddings_frame(which=which)
    ef = ef[(ef.block == block) & (ef.video_geo == region)]
    systems, item_keys = _rdm_vectors(ef)
    n_items = len(item_keys)
    humans = [s for s in systems if system_type(s).startswith("Human")]
    vlms = [s for s in systems if system_type(s) == "VLM"]
    full = np.arange(n_items)
    sim = {s: _full_sim(systems[s][0]) for s in systems}   # precomputed per system

    def stats_for(idx):
        ii, jj = _valid_pairs(idx)
        Hu = {h: _upper(sim[h], idx, ii, jj) for h in humans}
        Vu = {v: _upper(sim[v], idx, ii, jj) for v in vlms}
        Hmat = np.vstack(list(Hu.values()))
        mean_H = Hmat.mean(axis=0)
        # noise ceiling
        upper = np.mean([_corr(Hu[h], mean_H) for h in humans])
        lower = np.mean([
            _corr(Hu[h], (Hmat.sum(0) - Hu[h]) / (len(humans) - 1)) for h in humans
        ])
        # alignment of VLMs to human consensus
        vlm_to_H = np.mean([_corr(Vu[v], mean_H) for v in vlms])
        # human-human and vlm-vlm mean pairwise RSA
        def mean_pair(us):
            ks = list(us); c = []
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    c.append(_corr(us[ks[i]], us[ks[j]]))
            return float(np.mean(c))
        hh = mean_pair(Hu)
        vv = mean_pair(Vu)
        return upper, lower, vlm_to_H, hh, vv

    obs = stats_for(full)
    boots = np.empty((n_boot, 5))
    for b in range(n_boot):
        idx = RNG.choice(full, n_items, replace=True)
        boots[b] = stats_for(idx)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    names = ["nc_upper", "nc_lower", "vlm_to_human", "human_human", "vlm_vlm"]
    row = {"block": block, "block_name": BLOCK_NAME[block], "region": region,
           "n_items": n_items, "n_humans": len(humans), "n_vlms": len(vlms),
           "embedding": which}
    for k, name in enumerate(names):
        row[name] = float(obs[k])
        row[f"{name}_lo"] = float(lo[k])
        row[f"{name}_hi"] = float(hi[k])
    # is VLM-human alignment below the human floor?
    row["vlm_below_ceiling_floor"] = bool(obs[2] < obs[1])
    return row


def main(which="mpnet", n_boot=N_BOOT):
    rows = []
    for block in [1, 2, 3, 4]:
        for region in ["Lima", "NYC"]:
            rows.append(block_region_stats(block, region, which, n_boot))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, f"rsa_noise_ceiling_{which}.csv"), index=False)
    with open(os.path.join(OUTDIR, f"rsa_noise_ceiling_{which}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote rsa_noise_ceiling_{which}.csv")
    show = df[["block_name", "region", "nc_lower", "nc_upper",
               "vlm_to_human", "vlm_to_human_lo", "vlm_to_human_hi",
               "human_human", "vlm_vlm", "vlm_below_ceiling_floor"]]
    print(show.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "mpnet"
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else N_BOOT
    main(which, nb)
