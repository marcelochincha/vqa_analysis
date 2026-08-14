"""#2 -- Power analysis / minimum detectable effect (MDE) for the geography claim.

Without this, "no geography effect" is indistinguishable from "no statistical power".
We answer: with n = 10 videos per geography, what size of geography main effect on
the ratings would we have detected 80% of the time?

Geography is a property of the *video*, so the replication unit is the video (10
Lima, 10 NYC), NOT the ~3000 individual answers -- treating answers as independent
would pseudo-replicate and inflate power (a naive parametric F gives ~25% type-I
error here). We therefore power the correct video-level test: collapse each video to
its mean rating across systems and questions and run a two-sample comparison of the
10 Lima vs 10 NYC video means, whose noise is the genuine between-video variability.

Method:
  1. Estimate the pooled within-region SD of the per-video mean ratings.
  2. For a grid of true shifts delta (extra rating points on NYC videos), simulate
     10 Lima + 10 NYC video means and run a two-sample t-test.
  3. Power(delta) = P(p < .05); MDE = delta at 80% power, in rating points and in a
     video-level Cohen's d. Contrast with the observed Lima-NYC gap.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from data_io import OUTDIR, load_ratings

RNG = np.random.default_rng(20260810)
N_SIM = 5000
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]


def video_level(r):
    """Per-video mean rating and the pooled within-region SD (H0-clean noise)."""
    vbar = r.groupby(["VIDEO", "video_geo"])["rating"].mean().reset_index()
    lima = vbar[vbar.video_geo == "Lima"]["rating"].to_numpy()
    nyc = vbar[vbar.video_geo == "NYC"]["rating"].to_numpy()
    n1, n2 = len(lima), len(nyc)
    sd_pooled = np.sqrt(((n1 - 1) * lima.var(ddof=1) + (n2 - 1) * nyc.var(ddof=1))
                        / (n1 + n2 - 2))
    return vbar, lima, nyc, float(sd_pooled), n1, n2


def main(n_sim=N_SIM):
    r = load_ratings(rep=1)
    vbar, lima, nyc, sd_pooled, n1, n2 = video_level(r)

    rows = []
    for delta in DELTAS:
        rej = 0
        for _ in range(n_sim):
            a = RNG.normal(0.0, sd_pooled, n1)
            b = RNG.normal(delta, sd_pooled, n2)
            _, p = stats.ttest_ind(a, b)
            rej += (p < 0.05)
        rows.append(dict(delta_points=delta,
                         delta_cohens_d=delta / sd_pooled,
                         power=float(rej / n_sim)))
    df = pd.DataFrame(rows)
    vc = {"sd_pooled_videomean": sd_pooled, "n_per_group": n1}

    # interpolate MDE at 80% power
    mde = np.nan
    for i in range(1, len(df)):
        if df.power.iloc[i - 1] < 0.8 <= df.power.iloc[i]:
            x0, x1 = df.delta_points.iloc[i - 1], df.delta_points.iloc[i]
            y0, y1 = df.power.iloc[i - 1], df.power.iloc[i]
            mde = x0 + (0.8 - y0) * (x1 - x0) / (y1 - y0)
            break

    observed_gap = abs(lima.mean() - nyc.mean())
    summary = dict(
        sd_pooled_videomean=sd_pooled,
        n_videos_per_group=n1,
        MDE_points_at_80pct=float(mde),
        MDE_cohens_d_at_80pct=float(mde / sd_pooled) if np.isfinite(mde) else None,
        observed_LimaNYC_gap_points=float(observed_gap),
        observed_gap_cohens_d=float(observed_gap / sd_pooled),
        n_sim=n_sim,
    )
    df.to_csv(os.path.join(OUTDIR, "power_curve.csv"), index=False)
    with open(os.path.join(OUTDIR, "power_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("power curve (video-level, n=10 per geography):")
    print(df.round(3).to_string(index=False))
    print(f"\npooled within-region SD of per-video means: {sd_pooled:.3f}")
    print(f"MDE @80% power : {mde:.3f} rating points "
          f"(Cohen's d = {mde / sd_pooled:.3f})")
    print(f"observed gap   : {observed_gap:.3f} points "
          f"(d = {observed_gap / sd_pooled:.3f})  -> below MDE")
    return summary


if __name__ == "__main__":
    import sys
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else N_SIM
    main(ns)
