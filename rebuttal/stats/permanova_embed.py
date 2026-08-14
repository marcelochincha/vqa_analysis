"""#1c -- PERMANOVA: the inferential version of the PCA / cosine-heatmap claims.

The paper argues from *visual* inspection of PCA scatters and cosine heatmaps that
(a) system_type (Human vs VLM) separates answers while (b) video geography does not.
We test both with a distance-based factorial PERMANOVA (McArdle & Anderson 2001),
which yields a pseudo-F, a permutation p-value, and -- crucially -- R^2 as an
effect size for every term.

Objects: one centroid per (system, region) in a given block  -> 60 points
         (30 systems x {Lima, NYC}), each the mean answer embedding of that
         system over the videos/questions of that block-region. Using centroids
         (not individual answers) keeps the objects independent, so the
         permutation test is honest rather than pseudo-replicated.

Factors (fully crossed, balanced):
    system_type in {Human_Lima, Human_NYC, VLM}   (10 systems each)
    video_geo   in {Lima, NYC}

Permutation schemes match each null:
    geo         : within each system, randomly swap its two region labels.
    system_type : relabel the 30 systems' type (both region points move together).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data_io import OUTDIR, embeddings_frame, BLOCK_NAME, system_type

RNG = np.random.default_rng(20260810)
N_PERM = 5_000


# --------------------------------------------------------------- distance algebra
def gower(D2: np.ndarray) -> np.ndarray:
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ D2 @ J


def hat(X: np.ndarray) -> np.ndarray:
    return X @ np.linalg.pinv(X.T @ X) @ X.T


def _dum(labels):
    return pd.get_dummies(pd.Categorical(labels), drop_first=True).to_numpy(float)


def _inter(A, B):
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.empty((A.shape[0], 0))
    return np.hstack([A[:, i:i+1]*B[:, j:j+1]
                      for i in range(A.shape[1]) for j in range(B.shape[1])])


def permanova_terms(G, geo, sys):
    """McArdle-Anderson pseudo-F / R^2 for geo, system, interaction.

    Main effects use their marginal SS = tr(H[1,X] G H[1,X]) (>=0); the interaction
    uses the SS it adds on top of the two mains, SS_full - SS_main (>=0). This avoids
    the small negative pseudo-F that a naive full-minus-mains split can produce when
    the Gower space is not perfectly orthogonal.
    """
    n = G.shape[0]
    one = np.ones((n, 1))
    Gd = _dum(geo); Sd = _dum(sys); GS = _inter(Gd, Sd)
    tr = lambda H: float(np.trace(H @ G @ H))
    SS_total = float(np.trace(G))
    SS_geo = tr(hat(np.hstack([one, Gd])))            # marginal
    SS_sys = tr(hat(np.hstack([one, Sd])))            # marginal
    SS_main = tr(hat(np.hstack([one, Gd, Sd])))
    SS_full = tr(hat(np.hstack([one, Gd, Sd, GS])))
    SS_int = max(SS_full - SS_main, 0.0)              # added last
    SS_res = max(SS_total - SS_full, 1e-12)
    df_geo, df_sys, df_int = Gd.shape[1], Sd.shape[1], GS.shape[1]
    df_res = n - (1 + df_geo + df_sys + df_int)
    F = lambda ss, df: (ss/df) / (SS_res/df_res)
    return dict(
        F_geo=F(SS_geo, df_geo), R2_geo=SS_geo/SS_total,
        F_sys=F(SS_sys, df_sys), R2_sys=SS_sys/SS_total,
        F_int=F(SS_int, df_int), R2_int=SS_int/SS_total,
        R2_res=SS_res/SS_total,
    )


def block_centroids(block, which="mpnet"):
    """(system, region) centroids for a block; returns coords + factor labels."""
    ef = embeddings_frame(which=which)
    ef = ef[ef.block == block]
    rows = []
    for (sysname, region), g in ef.groupby(["system", "video_geo"]):
        vecs = np.vstack(g["vec"].to_numpy())
        rows.append((sysname, region, system_type(sysname), vecs.mean(axis=0)))
    sysnames = [r[0] for r in rows]
    regions = np.array([r[1] for r in rows])
    stypes = np.array([r[2] for r in rows])
    X = np.vstack([r[3] for r in rows])
    return sysnames, regions, stypes, X


def pairwise_D2(X, metric="euclidean"):
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        return D ** 2
    # squared euclidean
    sq = np.sum(X**2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.maximum(D2, 0.0)


def run_block(block, which="mpnet", metric="cosine", n_perm=N_PERM):
    sysnames, regions, stypes, X = block_centroids(block, which)
    G = gower(pairwise_D2(X, metric))
    obs = permanova_terms(G, regions, stypes)

    # index helpers for restricted permutation
    sys_of_obj = np.array(sysnames)
    uniq_sys = np.array(sorted(set(sysnames)))
    stype_of_sys = {s: system_type(s) for s in uniq_sys}

    ge = np.empty(n_perm); ie = np.empty(n_perm); se = np.empty(n_perm)
    for i in range(n_perm):
        # geo null: swap the two region labels within each system independently
        perm_geo = regions.copy()
        for s in uniq_sys:
            idx = np.where(sys_of_obj == s)[0]
            if len(idx) == 2 and RNG.random() < 0.5:
                perm_geo[idx] = perm_geo[idx][::-1]
        stg = permanova_terms(G, perm_geo, stypes)
        ge[i] = stg["F_geo"]; ie[i] = stg["F_int"]
        # system null: relabel system_type across systems (both points move together)
        shuffled = RNG.permutation([stype_of_sys[s] for s in uniq_sys])
        smap = dict(zip(uniq_sys, shuffled))
        perm_sys = np.array([smap[s] for s in sys_of_obj])
        se[i] = permanova_terms(G, regions, perm_sys)["F_sys"]

    pval = lambda null, o: float((1 + np.sum(null >= o)) / (1 + n_perm))
    return dict(
        block=block, block_name=BLOCK_NAME[block], embedding=which, metric=metric,
        n_objects=len(sysnames),
        F_geo=obs["F_geo"], R2_geo=obs["R2_geo"], p_geo=pval(ge, obs["F_geo"]),
        F_sys=obs["F_sys"], R2_sys=obs["R2_sys"], p_sys=pval(se, obs["F_sys"]),
        F_int=obs["F_int"], R2_int=obs["R2_int"], p_int=pval(ie, obs["F_int"]),
    )


def main(which="mpnet", metric="cosine", n_perm=N_PERM):
    rows = [run_block(b, which, metric, n_perm) for b in [1, 2, 3, 4]]
    df = pd.DataFrame(rows)
    tag = f"{which}_{metric}"
    df.to_csv(os.path.join(OUTDIR, f"permanova_{tag}.csv"), index=False)
    with open(os.path.join(OUTDIR, f"permanova_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote permanova_{tag}.csv")
    show = df[["block_name", "F_geo", "R2_geo", "p_geo",
               "F_sys", "R2_sys", "p_sys", "F_int", "R2_int", "p_int"]]
    print(show.round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "mpnet"
    metric = sys.argv[2] if len(sys.argv) > 2 else "cosine"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else N_PERM
    main(which, metric, n)
