"""#1a / #1b / #13 -- Inferential test of the 2x3 factorial on Block-2 ratings.

The paper calls itself a "fully factorial 2x3 design" but never fits a factorial
model. This module supplies:

  * A distribution-free PERMUTATION test for the geography main effect and the
    geography x system_type interaction -- the reviewer's literal request
    ("shuffle geography labels").  Geography is a property of the video, so the
    null is built by reshuffling the 10/10 Lima/NYC assignment of the 20 clips.
  * A partial-F statistic (type-II-style) computed with plain numpy lstsq, so the
    10k-permutation loop stays fast on a CPU.
  * Parametric back-ups: a linear mixed model (random intercepts for subject and
    video) and an ordinal model, whose geo p-values should agree with the
    permutation p.
  * Effect sizes with CIs: partial eta^2 for each term, the Lima-NYC mean gap with
    a cluster bootstrap CI, Cohen's d for Human-vs-VLM and Human_Lima-vs-Human_NYC.
  * #13: the whole thing re-run on per-subject z-scored ratings (each participant
    centred/scaled by their own mean/sd) so ordinal-scale-use differences cannot
    manufacture or hide an effect.

Everything is read-only; results are written to rebuttal/outputs/.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data_io import OUTDIR, load_ratings, BLOCK_NAME

RNG = np.random.default_rng(20260810)
N_PERM = 10_000
N_BOOT = 10_000


# --------------------------------------------------------------- linear algebra
def _ssr(y: np.ndarray, X: np.ndarray) -> float:
    """Residual sum of squares of OLS y ~ X (X already includes intercept)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def partial_F(y, X_full, X_red):
    """Partial-F for the columns present in X_full but not X_red."""
    ssr_f = _ssr(y, X_full)
    ssr_r = _ssr(y, X_red)
    df_diff = X_full.shape[1] - X_red.shape[1]
    df_res = X_full.shape[0] - X_full.shape[1]
    if ssr_f <= 0 or df_diff <= 0 or df_res <= 0:
        return np.nan, np.nan
    F = ((ssr_r - ssr_f) / df_diff) / (ssr_f / df_res)
    partial_eta2 = (ssr_r - ssr_f) / ssr_r  # SS_effect / (SS_effect + SS_error_reduced)
    return F, partial_eta2


def _dummies(labels: np.ndarray) -> np.ndarray:
    """Reduced dummy coding (drop first level) without intercept column."""
    cats = pd.Categorical(labels)
    D = pd.get_dummies(cats, drop_first=True).to_numpy(dtype=np.float64)
    return D


def _interaction(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """All pairwise products of the columns of A and B."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.empty((A.shape[0], 0))
    cols = [A[:, i:i + 1] * B[:, j:j + 1] for i in range(A.shape[1]) for j in range(B.shape[1])]
    return np.hstack(cols)


def _design(geo, sys, q):
    """Return the design-matrix pieces for the factorial model."""
    n = len(geo)
    one = np.ones((n, 1))
    G = _dummies(geo)
    S = _dummies(sys)
    Q = _dummies(q)
    GS = _interaction(G, S)
    return one, G, S, Q, GS


# --------------------------------------------------------------- statistics
def geo_stats(df: pd.DataFrame):
    """F / partial-eta^2 for geo main effect, system main effect, interaction."""
    y = df["y"].to_numpy(dtype=np.float64)
    one, G, S, Q, GS = _design(df["video_geo"].values, df["system_type"].values,
                               df["QUESTION_NUM"].values)
    base = np.hstack([one, Q])              # intercept + question blocking
    m_sys = np.hstack([base, S])           # + system
    m_geo = np.hstack([base, G])           # + geo
    m_main = np.hstack([base, G, S])       # geo + system
    m_full = np.hstack([base, G, S, GS])   # + interaction

    F_geo, eta_geo = partial_F(y, m_main, m_sys)        # geo | system,q
    F_sys, eta_sys = partial_F(y, m_main, m_geo)        # system | geo,q
    F_int, eta_int = partial_F(y, m_full, m_main)       # interaction | mains
    return dict(F_geo=F_geo, eta2_geo=eta_geo,
                F_sys=F_sys, eta2_sys=eta_sys,
                F_int=F_int, eta2_int=eta_int)


def _F_from_designs(y, m_full, m_red):
    F, _ = partial_F(y, m_full, m_red)
    return F


def perm_test(df: pd.DataFrame, n_perm=N_PERM):
    """Permutation p-values (precomputed designs; numpy-only shuffling).

    geo terms  : reshuffle the video->geo map (keeps 10/10) -> breaks any geo
                 association while preserving each system's rating distribution.
    system term: reshuffle the system_type label across the 30 systems.
    """
    obs = geo_stats(df)
    y = df["y"].to_numpy(dtype=np.float64)
    n = len(y)
    one = np.ones((n, 1))
    S = _dummies(df["system_type"].values)      # constant under geo perm
    Q = _dummies(df["QUESTION_NUM"].values)      # constant everywhere
    base = np.hstack([one, Q])
    base_S = np.hstack([base, S])                # geo-null reduced model (F_geo)
    base_G = np.hstack([base, _dummies(df["video_geo"].values)])  # sys-null red (F_sys)

    # index maps so we rebuild dummy columns by integer indexing, not pandas
    videos = df["VIDEO"].to_numpy()
    uniq_vid = np.array(sorted(df["VIDEO"].unique()))
    n_lima = int((df.drop_duplicates("VIDEO")["video_geo"] == "Lima").sum())
    agents = df["AGENT"].to_numpy()
    uniq_ag = np.array(sorted(df["AGENT"].unique()))
    ag_idx = {a: i for i, a in enumerate(uniq_ag)}
    row_ag = np.array([ag_idx[a] for a in agents])
    sys_levels = np.array(sorted(df["system_type"].unique()))     # 3 levels
    sys_of_ag = df.drop_duplicates("AGENT").set_index("AGENT")["system_type"]
    ag_sys = np.array([sys_levels.tolist().index(sys_of_ag[a]) for a in uniq_ag])

    ge = np.zeros(n_perm); ie = np.zeros(n_perm); se = np.zeros(n_perm)
    for i in range(n_perm):
        # --- geo permutation: only G (1 col) and GS change
        lima_set = set(RNG.choice(len(uniq_vid), size=n_lima, replace=False).tolist())
        lima_vids = uniq_vid[list(lima_set)]
        Gp = np.isin(videos, lima_vids).astype(np.float64)[:, None]
        GSp = _interaction(Gp, S)
        m_main = np.hstack([base_S, Gp])          # base + S + G  == geo | sys,q
        m_full = np.hstack([m_main, GSp])
        ge[i] = _F_from_designs(y, m_main, base_S)
        ie[i] = _F_from_designs(y, m_full, m_main)
        # --- system permutation: relabel the 3 system_type levels over agents
        perm = RNG.permutation(ag_sys)
        row_sys = perm[row_ag]                     # per-row permuted level index
        Sp = pd.get_dummies(pd.Categorical(row_sys, categories=[0, 1, 2]),
                            drop_first=True).to_numpy(dtype=np.float64)
        m_main_s = np.hstack([base_G, Sp])         # base + G + S == sys | geo,q
        se[i] = _F_from_designs(y, m_main_s, base_G)

    def pval(null, obs_val):
        null = null[~np.isnan(null)]
        return float((1 + np.sum(null >= obs_val)) / (1 + len(null)))

    return dict(
        p_geo=pval(ge, obs["F_geo"]),
        p_int=pval(ie, obs["F_int"]),
        p_sys=pval(se, obs["F_sys"]),
        **obs,
    )


def cluster_bootstrap_gap(df, n_boot=N_BOOT):
    """Lima-NYC mean-rating gap with a video-cluster bootstrap 95% CI."""
    lima_v = df[df.video_geo == "Lima"].VIDEO.unique()
    nyc_v = df[df.video_geo == "NYC"].VIDEO.unique()
    by_vid = {v: df[df.VIDEO == v]["y"].to_numpy() for v in np.r_[lima_v, nyc_v]}
    obs = df[df.video_geo == "Lima"].y.mean() - df[df.video_geo == "NYC"].y.mean()
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        lv = RNG.choice(lima_v, size=len(lima_v), replace=True)
        nv = RNG.choice(nyc_v, size=len(nyc_v), replace=True)
        lm = np.concatenate([by_vid[v] for v in lv]).mean()
        nm = np.concatenate([by_vid[v] for v in nv]).mean()
        gaps[b] = lm - nm
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def contrast_d_ci(df, group_a, group_b, n_boot=N_BOOT):
    """Cohen's d between two system_type groups, bootstrap CI over subjects."""
    A = df[df.system_type == group_a]
    B = df[df.system_type == group_b]
    ag = A.AGENT.unique(); bg = B.AGENT.unique()
    by = {k: df[df.AGENT == k]["y"].to_numpy() for k in np.r_[ag, bg]}
    obs = cohens_d(A.y, B.y)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        aa = np.concatenate([by[k] for k in RNG.choice(ag, len(ag), replace=True)])
        bb = np.concatenate([by[k] for k in RNG.choice(bg, len(bg), replace=True)])
        ds[i] = cohens_d(aa, bb)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return obs, float(lo), float(hi)


# --------------------------------------------------------------- parametric back-ups
def parametric_backups(df):
    """Mixed model (subject+video random intercepts) and ordinal model geo p-values."""
    import warnings
    import statsmodels.formula.api as smf
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    d = df.rename(columns={"y": "rating"}).copy()
    d["QUESTION_NUM"] = d["QUESTION_NUM"].astype(str)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.mixedlm(
                "rating ~ C(video_geo)*C(system_type) + C(QUESTION_NUM)",
                d, groups=d["AGENT"], vc_formula={"video": "0 + C(VIDEO)"},
            ).fit(reml=False, method="lbfgs")
            geo_p = {k: float(v) for k, v in m.pvalues.items() if "video_geo" in k}
            out["mixedlm_geo_pvalues"] = geo_p
        except Exception as e:  # pragma: no cover
            out["mixedlm_error"] = str(e)
        try:
            X = pd.get_dummies(
                d[["video_geo", "system_type", "QUESTION_NUM"]], drop_first=True
            ).astype(float)
            om = OrderedModel(d["rating"], X, distr="logit").fit(method="bfgs", disp=False)
            out["ordinal_geo_pvalues"] = {
                k: float(v) for k, v in om.pvalues.items() if "video_geo" in k
            }
        except Exception as e:  # pragma: no cover
            out["ordinal_error"] = str(e)
    return out


# --------------------------------------------------------------- driver
def _persubject_z(df):
    d = df.copy()
    g = d.groupby("AGENT")["y"]
    d["y"] = (d["y"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return d.dropna(subset=["y"])


def analyse(df, label, n_perm):
    res = {"label": label, "n": int(len(df))}
    res.update(perm_test(df, n_perm))
    gap, lo, hi = cluster_bootstrap_gap(df)
    res["gap_LimaNYC"] = gap
    res["gap_CI"] = [lo, hi]
    for a, b, key in [("Human_Lima", "VLM", "d_HumanLima_vs_VLM"),
                      ("Human_NYC", "VLM", "d_HumanNYC_vs_VLM"),
                      ("Human_Lima", "Human_NYC", "d_HumanLima_vs_HumanNYC")]:
        d, l, h = contrast_d_ci(df, a, b)
        res[key] = {"d": d, "CI": [l, h]}
    return res


def main(n_perm=N_PERM):
    ratings = load_ratings(rep=1)
    ratings = ratings.rename(columns={"rating": "y"})

    report = {"design": "2x3 (video_geo x system_type)", "rep": 1,
              "n_perm": n_perm, "measure": "Block-2 Likert ratings 1-10",
              "blocks": {}}

    # ----- overall Block 2, raw + per-subject normalised
    report["block2_overall_raw"] = analyse(ratings, "Block2_raw", n_perm)
    report["block2_overall_zscored"] = analyse(_persubject_z(ratings),
                                               "Block2_persubject_z", n_perm)
    report["parametric_backups"] = parametric_backups(ratings)

    # ----- per question (Q6-Q10)
    perq = {}
    for q in sorted(ratings.QUESTION_NUM.unique()):
        sub = ratings[ratings.QUESTION_NUM == q]
        perq[f"Q{q}"] = analyse(sub, f"Q{q}", n_perm)
    report["per_question"] = perq

    with open(os.path.join(OUTDIR, "factorial_ratings.json"), "w") as f:
        json.dump(report, f, indent=2)

    # flat CSV for the paper table
    rows = []
    for key in ["block2_overall_raw", "block2_overall_zscored"]:
        r = report[key]
        rows.append({
            "analysis": key, "n": r["n"],
            "F_geo": r["F_geo"], "p_geo": r["p_geo"], "eta2_geo": r["eta2_geo"],
            "F_sys": r["F_sys"], "p_sys": r["p_sys"], "eta2_sys": r["eta2_sys"],
            "F_int": r["F_int"], "p_int": r["p_int"], "eta2_int": r["eta2_int"],
            "gap_LimaNYC": r["gap_LimaNYC"], "gap_lo": r["gap_CI"][0], "gap_hi": r["gap_CI"][1],
            "d_HL_vs_VLM": r["d_HumanLima_vs_VLM"]["d"],
            "d_HN_vs_VLM": r["d_HumanNYC_vs_VLM"]["d"],
            "d_HL_vs_HN": r["d_HumanLima_vs_HumanNYC"]["d"],
        })
    for q, r in report["per_question"].items():
        rows.append({
            "analysis": q, "n": r["n"],
            "F_geo": r["F_geo"], "p_geo": r["p_geo"], "eta2_geo": r["eta2_geo"],
            "F_sys": r["F_sys"], "p_sys": r["p_sys"], "eta2_sys": r["eta2_sys"],
            "F_int": r["F_int"], "p_int": r["p_int"], "eta2_int": r["eta2_int"],
            "gap_LimaNYC": r["gap_LimaNYC"], "gap_lo": r["gap_CI"][0], "gap_hi": r["gap_CI"][1],
            "d_HL_vs_VLM": r["d_HumanLima_vs_VLM"]["d"],
            "d_HN_vs_VLM": r["d_HumanNYC_vs_VLM"]["d"],
            "d_HL_vs_HN": r["d_HumanLima_vs_HumanNYC"]["d"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "factorial_ratings.csv"), index=False)
    print("wrote factorial_ratings.{json,csv} to", OUTDIR)
    r = report["block2_overall_raw"]
    print(f"\nGEO main effect : F={r['F_geo']:.3f}  p_perm={r['p_geo']:.4f}  "
          f"partial_eta2={r['eta2_geo']:.4f}")
    print(f"SYS main effect : F={r['F_sys']:.3f}  p_perm={r['p_sys']:.4f}  "
          f"partial_eta2={r['eta2_sys']:.4f}")
    print(f"GEOxSYS interact: F={r['F_int']:.3f}  p_perm={r['p_int']:.4f}  "
          f"partial_eta2={r['eta2_int']:.4f}")
    print(f"Lima-NYC gap    : {r['gap_LimaNYC']:+.3f}  CI{r['gap_CI']}")
    print(f"d Human_Lima vs VLM : {r['d_HumanLima_vs_VLM']}")
    print(f"d Human_Lima vs Human_NYC : {r['d_HumanLima_vs_HumanNYC']}")
    return report


if __name__ == "__main__":
    import sys
    np_ = int(sys.argv[1]) if len(sys.argv) > 1 else N_PERM
    main(np_)
