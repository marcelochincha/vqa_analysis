"""Build the supplementary figure + LaTeX tables for the repetition analysis.

Reads the JSON produced by repetition_consistency.py and
factorial_repmean_robustness.py and writes, to rebuttal/outputs/:
  repetition_consistency.pdf / .png     -- 2-panel figure
  repetition_consistency.tex            -- summary table (numeric + semantic)
  factorial_repmean_robustness.tex      -- rep1-vs-rep-mean robustness table
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_io import OUTDIR

J = json.load(open(os.path.join(OUTDIR, "repetition_consistency.json")))
NUM, SEM = J["numeric_block2"], J["semantic_freetext"]
ROB = json.load(open(os.path.join(OUTDIR, "factorial_repmean_robustness.json")))

CLOSED = {"Gemini3-Flash-preview", "Gemini3-Pro-preview"}
DISP = {"Gemini3-Flash-preview": "Gemini-3-Flash", "Gemini3-Pro-preview": "Gemini-3-Pro",
        "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B", "Cosmos-Reason2-8B": "Cosmos-Reason2-8B",
        "MiniCPM-o-2_6": "MiniCPM-o-2.6", "LLaVA-Video-7B-Qwen2": "LLaVA-Video-7B",
        "InternVL3-8B": "InternVL3-8B", "Perception-LM-8B": "Perception-LM-8B",
        "Phi-4-multimodal-instruct": "Phi-4-MM", "VideoLLaMA3-7B": "VideoLLaMA3-7B"}


# ---------------------------------------------------------------- figure
def figure():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left: semantic self vs between-VLM by block
    blocks = list(SEM["by_block"].keys())
    self_v = [SEM["by_block"][b]["self"] for b in blocks]
    betw_v = [SEM["by_block"][b]["between_vlm"] for b in blocks]
    x = np.arange(len(blocks)); w = 0.38
    axL.bar(x - w / 2, self_v, w, label="within-VLM (20 reps)", color="#2c7fb8")
    axL.bar(x + w / 2, betw_v, w, label="between distinct VLMs", color="#c0c0c0")
    for xi, v in zip(x - w / 2, self_v):
        axL.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, betw_v):
        axL.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels(blocks)
    axL.set_ylabel("mean pairwise cosine")
    axL.set_ylim(0, 1.0)
    axL.set_title("(a) Free-text self-consistency\nrepetitions cluster tighter than systems")
    axL.legend(fontsize=8, loc="lower right")

    # --- Right: per-VLM numeric within-rep SD
    items = sorted(NUM["by_vlm"].items(), key=lambda kv: kv[1])
    names = [DISP.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d95f02" if k in CLOSED else "#7570b3" for k, _ in items]
    y = np.arange(len(names))
    axR.barh(y, vals, color=colors)
    axR.axvline(NUM["between_video_signal_sd"], ls="--", color="k", lw=1)
    axR.text(NUM["between_video_signal_sd"], len(names) - 0.4,
             f" between-video\n signal SD={NUM['between_video_signal_sd']:.2f}",
             fontsize=7, va="top")
    axR.set_yticks(y); axR.set_yticklabels(names, fontsize=8)
    axR.set_xlabel("within-VLM SD across 20 reps (rating pts)")
    axR.set_title("(b) Block-2 rating stochasticity\n(orange = Gemini T=1.0, purple = open T=0.5)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"repetition_consistency.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote repetition_consistency.pdf/.png")


# ---------------------------------------------------------------- tables
def summary_tex():
    g = SEM["geo_self_cosine"]; gn = NUM["geo_rep_sd"]
    rows = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\multicolumn{4}{l}{\textbf{(a) Free-text semantic self-consistency (mean pairwise cosine)}}\\",
        r"\midrule",
        r"Block & within-VLM (20 reps) & between distinct VLMs & --\\",
        r"\midrule",
    ]
    for b, d in SEM["by_block"].items():
        rows.append(f"{b} & {d['self']:.3f} & {d['between_vlm']:.3f} & \\\\")
    rows += [
        r"\midrule",
        f"All free-text & {SEM['mean_self_cosine']:.3f} & {SEM['mean_between_vlm_cosine']:.3f} & \\\\",
        f"Lima vs NYC & {g['Lima']:.3f} / {g['NYC']:.3f} & "
        f"\\multicolumn{{2}}{{l}}{{perm.\\ $p={g['perm_p']:.2f}$ (no geo.\\ effect)}}\\\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textbf{(b) Block-2 numeric run-to-run stochasticity (rating points, 1--10)}}\\",
        r"\midrule",
        f"Mean within-VLM rep.\\ SD & \\multicolumn{{3}}{{l}}{{{NUM['mean_rep_sd']:.2f}}}\\\\",
        f"Between-video signal SD & \\multicolumn{{3}}{{l}}{{{NUM['between_video_signal_sd']:.2f}}}\\\\",
        f"Gemini (T=1.0) vs open (T=0.5) & \\multicolumn{{3}}{{l}}{{"
        f"{NUM['closed_vs_open_rep_sd']['closed_T1.0']:.2f} vs "
        f"{NUM['closed_vs_open_rep_sd']['open_T0.5']:.2f}}}\\\\",
        f"Lima vs NYC rep.\\ SD & \\multicolumn{{3}}{{l}}{{{gn['Lima']:.2f} / {gn['NYC']:.2f}, "
        f"perm.\\ $p={gn['perm_p']:.2f}$}}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(rows)
    open(os.path.join(OUTDIR, "repetition_consistency.tex"), "w").write(tex)
    print("wrote repetition_consistency.tex")


def robustness_tex():
    r1, rm = ROB["rep1"], ROB["rep_mean20"]
    def cell(v, fmt, is_p):
        return r"$<0.001$" if is_p and isinstance(v, float) and 0 <= v < 0.001 else fmt.format(v)
    def row(name, a, b, fmt="{:.3f}", is_p=False):
        return f"{name} & {cell(a, fmt, is_p)} & {cell(b, fmt, is_p)}\\\\"
    rows = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Block-2 factorial term & rep.\ 1 (paper) & mean of 20 reps\\",
        r"\midrule",
        row(r"Geography $p_{\text{perm}}$", r1["p_geo"], rm["p_geo"], is_p=True),
        row(r"Geography partial $\eta^2$", r1["eta2_geo"], rm["eta2_geo"], "{:.4f}"),
        row(r"System $p_{\text{perm}}$", r1["p_sys"], rm["p_sys"], is_p=True),
        row(r"System partial $\eta^2$", r1["eta2_sys"], rm["eta2_sys"]),
        row(r"Geo.$\times$System $p_{\text{perm}}$", r1["p_int"], rm["p_int"], is_p=True),
        row(r"Lima--NYC gap (pts)", r1["gap_LimaNYC"], rm["gap_LimaNYC"]),
        row(r"Cohen's $d$ Human--VLM", r1["d_HumanLima_vs_VLM"]["d"], rm["d_HumanLima_vs_VLM"]["d"]),
        row(r"Cohen's $d$ Lima--NYC humans", r1["d_HumanLima_vs_HumanNYC"]["d"], rm["d_HumanLima_vs_HumanNYC"]["d"]),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex = "\n".join(rows)
    open(os.path.join(OUTDIR, "factorial_repmean_robustness.tex"), "w").write(tex)
    print("wrote factorial_repmean_robustness.tex")


if __name__ == "__main__":
    figure()
    summary_tex()
    robustness_tex()
    print("\n--- markdown preview (summary) ---")
    print(f"Free-text self vs between-VLM (all): {SEM['mean_self_cosine']} vs {SEM['mean_between_vlm_cosine']}")
    for b, d in SEM["by_block"].items():
        print(f"  {b:15s} self={d['self']}  between={d['between_vlm']}")
    print(f"Numeric rep SD={NUM['mean_rep_sd']:.2f}  signal SD={NUM['between_video_signal_sd']:.2f} "
          f"| Gemini {NUM['closed_vs_open_rep_sd']['closed_T1.0']} vs open {NUM['closed_vs_open_rep_sd']['open_T0.5']}")
    print(f"Geo nulls: self-cos p={SEM['geo_self_cosine']['perm_p']}  rep-SD p={NUM['geo_rep_sd']['perm_p']}")
