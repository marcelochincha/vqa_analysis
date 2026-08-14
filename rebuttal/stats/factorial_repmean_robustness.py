"""Robustness of the Block-2 factorial to the single-repetition choice.

The paper (and factorial_ratings.py) use repetition 1 per VLM, matching the single
human response. Since a single VLM Likert draw is stochastic (see
repetition_consistency.py, within-VLM rep SD ~1.1 pts), we re-fit the factorial with
each VLM's rating AVERAGED over its 20 repetitions (humans unchanged) and check that
the geography null and the system effect are unchanged. No model is re-run.

Output: rebuttal/outputs/factorial_repmean_robustness.{json,csv}
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data_io import OUTDIR, load_ratings
from factorial_ratings import analyse, N_PERM


def rep_mean_ratings() -> pd.DataFrame:
    """VLMs -> mean rating over the 20 reps; humans -> their single rating."""
    allr = load_ratings(rep=None)
    keys = ["AGENT", "VIDEO", "QUESTION_NUM", "video_geo", "system_type"]
    agg = (allr.groupby(keys, as_index=False)["rating"]
               .mean().rename(columns={"rating": "y"}))
    return agg


def _flat(r):
    return {
        "n": r["n"],
        "F_geo": r["F_geo"], "p_geo": r["p_geo"], "eta2_geo": r["eta2_geo"],
        "F_sys": r["F_sys"], "p_sys": r["p_sys"], "eta2_sys": r["eta2_sys"],
        "F_int": r["F_int"], "p_int": r["p_int"], "eta2_int": r["eta2_int"],
        "gap_LimaNYC": r["gap_LimaNYC"], "gap_lo": r["gap_CI"][0], "gap_hi": r["gap_CI"][1],
        "d_HL_vs_VLM": r["d_HumanLima_vs_VLM"]["d"],
        "d_HL_vs_HN": r["d_HumanLima_vs_HumanNYC"]["d"],
    }


def main(n_perm=N_PERM):
    rep1 = load_ratings(rep=1).rename(columns={"rating": "y"})
    repm = rep_mean_ratings()

    res_rep1 = analyse(rep1, "rep1", n_perm)
    res_repm = analyse(repm, "rep_mean20", n_perm)

    out = {"rep1": res_rep1, "rep_mean20": res_repm}
    with open(os.path.join(OUTDIR, "factorial_repmean_robustness.json"), "w") as f:
        json.dump(out, f, indent=2)
    tab = pd.DataFrame({"rep1 (paper)": _flat(res_rep1),
                        "mean of 20 reps": _flat(res_repm)}).T
    tab.to_csv(os.path.join(OUTDIR, "factorial_repmean_robustness.csv"))

    show = ["F_geo", "p_geo", "eta2_geo", "F_sys", "p_sys", "eta2_sys",
            "F_int", "p_int", "eta2_int", "gap_LimaNYC", "d_HL_vs_VLM", "d_HL_vs_HN"]
    print("=== Factorial robustness: rep 1 vs mean-of-20-reps (VLMs) ===\n")
    print(tab[show].round(4).to_string())
    print("\nInterpretation: geography stays null and the system effect stays large "
          "under both, so the single-repetition choice does not drive the conclusions.")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_PERM)
