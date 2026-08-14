"""Judge-vs-human agreement: a tiny local GUI to hand-label a small pair sample,
then compute Cohen's kappa between YOUR labels and the LLM judge (#10 / Q6).

No GPU, no model re-run. Three subcommands (all local):

  make-sheet : draw a stratified subsample of pairs from the judge parquet and write
               `judge_human_sheet.csv` (with the judge's own verdict kept in hidden
               columns so we can join later; the GUI never shows them).
                 python judge_human_gui.py make-sheet --n-per-block 25

  gui        : open the annotation window. Shows the question + the two answers, you
               click Stage-1 (comparable?) then Stage-2 (agreement level), with a
               progress bar. Auto-saves after every decision and resumes where you
               left off. This is the default if no subcommand is given.
                 python judge_human_gui.py

  kappa      : after you finish, compute judge-vs-human Cohen's kappa (Stage-1 binary,
               Stage-2 nominal + quadratic-weighted, and collapsed agree/disagree).
                 python judge_human_gui.py kappa

The two answers are shown WITHOUT revealing which system produced them or what the
judge decided, so your labels are unbiased.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUTDIR, exist_ok=True)
PARQUET = os.path.join(
    REPO, "outputs", "pipeline_non_embed", "judge", "llm_agreement_scores.parquet")
SHEET = os.path.join(OUTDIR, "judge_human_sheet.csv")
BLOCK_NAME = {1: "Factual", 2: "Ratings", 3: "Counterfactual", 4: "Reasoning"}

# columns we display to the annotator (judge verdict deliberately excluded)
JUDGE_COLS = ["JUDGE_STAGE1", "JUDGE_FINAL"]
HUMAN_COLS = ["HUMAN_COMPARABLE", "HUMAN_SCORE"]

STAGE2_HINT = (
    "+2  Acuerdo fuerte: misma conclusion central (uno puede ser mas especifico).\n"
    "+1  Acuerdo parcial: concuerdan en lo central, pero uno agrega un dato extra.\n"
    "-1  Contradiccion parcial: mismo objeto/accion, distinto grado o intensidad.\n"
    "-2  Contradiccion directa: afirmaciones logicamente opuestas."
)


# --------------------------------------------------------------- make-sheet
def cmd_make_sheet(args):
    cols = ["VIDEO", "QUESTION_NUM", "QUESTION", "VIDEO_SECTOR", "BLOCK",
            "AGENT_I", "ANSWER_I", "AGENT_J", "ANSWER_J",
            "STAGE1_SCORE", "FINAL_SCORE"]
    d = pd.read_parquet(PARQUET, columns=cols)          # memory-safe: no text blobs
    d = d[d.BLOCK.isin([1, 3, 4])].copy()               # judge does not score numeric B2
    for c in ("ANSWER_I", "ANSWER_J"):
        d = d[d[c].astype(str).str.strip().str.len() > 0]
    rng = np.random.default_rng(args.seed)
    keep = []
    for b, g in d.groupby("BLOCK"):
        take = min(args.n_per_block, len(g))
        keep.append(g.loc[rng.choice(g.index.to_numpy(), take, replace=False)])
    s = pd.concat(keep).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out = pd.DataFrame({
        "PAIR_ID": np.arange(len(s)),
        "BLOCK": s.BLOCK.to_numpy(),
        "BLOCK_NAME": s.BLOCK.map(BLOCK_NAME).to_numpy(),
        "QUESTION": s.QUESTION.to_numpy(),
        "ANSWER_A": s.ANSWER_I.to_numpy(),
        "ANSWER_B": s.ANSWER_J.to_numpy(),
        # hidden reference (never shown in the GUI) --------------------------
        "AGENT_A": s.AGENT_I.to_numpy(),
        "AGENT_B": s.AGENT_J.to_numpy(),
        "JUDGE_STAGE1": s.STAGE1_SCORE.to_numpy(),
        "JUDGE_FINAL": s.FINAL_SCORE.to_numpy(),
        # to be filled by you ------------------------------------------------
        "HUMAN_COMPARABLE": np.nan,
        "HUMAN_SCORE": np.nan,
    })
    if os.path.exists(SHEET) and not args.overwrite:
        raise SystemExit(f"{SHEET} already exists; pass --overwrite to replace it "
                         f"(this WIPES your labels).")
    out.to_csv(SHEET, index=False)
    print(f"wrote {SHEET}  ({len(out)} pairs, "
          f"blocks {out.BLOCK.value_counts().to_dict()})")
    print("Now run:  python judge_human_gui.py")


# --------------------------------------------------------------- gui
def cmd_gui(args):
    import tkinter as tk
    from tkinter import ttk, messagebox

    if not os.path.exists(SHEET):
        raise SystemExit(f"No sheet found at {SHEET}. Run first:\n"
                         f"  python judge_human_gui.py make-sheet")
    df = pd.read_csv(SHEET)

    def save():
        df.to_csv(SHEET, index=False)

    def first_unlabeled():
        m = df["HUMAN_COMPARABLE"].isna()
        return int(m.idxmax()) if m.any() else len(df) - 1

    state = {"idx": first_unlabeled()}

    root = tk.Tk()
    root.title("Judge-vs-Human annotation")
    root.geometry("980x760")
    root.minsize(820, 640)

    # ---- header: progress ------------------------------------------------
    top = ttk.Frame(root, padding=10); top.pack(fill="x")
    prog = ttk.Progressbar(top, length=760, mode="determinate")
    prog.pack(side="left", padx=(0, 10))
    lbl_prog = ttk.Label(top, text=""); lbl_prog.pack(side="left")

    lbl_block = ttk.Label(root, text="", font=("Segoe UI", 11, "bold"))
    lbl_block.pack(anchor="w", padx=12)

    # ---- question --------------------------------------------------------
    qframe = ttk.LabelFrame(root, text="Pregunta", padding=8)
    qframe.pack(fill="x", padx=12, pady=(6, 4))
    txt_q = tk.Text(qframe, height=2, wrap="word", font=("Segoe UI", 11),
                    background="#f4f4f4", relief="flat")
    txt_q.pack(fill="x")

    # ---- two answers -----------------------------------------------------
    aframe = ttk.Frame(root); aframe.pack(fill="both", expand=True, padx=12, pady=4)
    aframe.columnconfigure(0, weight=1); aframe.columnconfigure(1, weight=1)
    aframe.rowconfigure(0, weight=1)

    def answer_box(parent, title, color):
        f = ttk.LabelFrame(parent, text=title, padding=6)
        t = tk.Text(f, wrap="word", font=("Segoe UI", 12), height=8,
                    background=color, relief="flat")
        t.pack(fill="both", expand=True)
        return f, t

    fa, txt_a = answer_box(aframe, "Respuesta A", "#eef6ff")
    fb, txt_b = answer_box(aframe, "Respuesta B", "#fff6ee")
    fa.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    fb.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    # ---- stage 1 ---------------------------------------------------------
    s1 = ttk.LabelFrame(
        root, text="Paso 1  --  Las dos respuestas, hablan de lo mismo? (se pueden comparar)",
        padding=8)
    s1.pack(fill="x", padx=12, pady=(6, 2))
    lbl_s1 = ttk.Label(s1, text="", font=("Segoe UI", 10, "italic"),
                       foreground="#0a7d24")
    lbl_s1.pack(anchor="w", pady=(0, 4))

    # ---- stage 2 ---------------------------------------------------------
    s2 = ttk.LabelFrame(root, text="Paso 2  --  Nivel de acuerdo", padding=8)
    s2.pack(fill="x", padx=12, pady=(2, 4))
    ttk.Label(s2, text=STAGE2_HINT, font=("Consolas", 9),
              foreground="#555").pack(anchor="w", pady=(0, 6))
    s2_btns = ttk.Frame(s2); s2_btns.pack(anchor="w")
    lbl_s2 = ttk.Label(s2, text="", font=("Segoe UI", 10, "italic"),
                       foreground="#0a4b7d")
    lbl_s2.pack(anchor="w", pady=(4, 0))

    # ---- nav -------------------------------------------------------------
    nav = ttk.Frame(root, padding=8); nav.pack(fill="x")
    btn_prev = ttk.Button(nav, text="<-- Anterior (Left)")
    btn_prev.pack(side="left")
    lbl_saved = ttk.Label(nav, text="", foreground="#0a7d24")
    lbl_saved.pack(side="left", padx=12)
    btn_next = ttk.Button(nav, text="Siguiente (Right) -->")
    btn_next.pack(side="right")

    # ------------------------------------------------------------------ render
    def set_text(widget, s):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", "" if pd.isna(s) else str(s))
        widget.configure(state="disabled")

    def render():
        i = state["idx"]
        row = df.iloc[i]
        n = len(df)
        done = int(df["HUMAN_COMPARABLE"].notna().sum())
        prog["maximum"] = n; prog["value"] = done
        lbl_prog.config(text=f"{done}/{n} etiquetados")
        lbl_block.config(text=f"Par {i + 1} / {n}   -   Bloque: {row['BLOCK_NAME']}")
        set_text(txt_q, row["QUESTION"])
        set_text(txt_a, row["ANSWER_A"])
        set_text(txt_b, row["ANSWER_B"])

        comp = row["HUMAN_COMPARABLE"]
        sc = row["HUMAN_SCORE"]
        if pd.isna(comp):
            lbl_s1.config(text="(sin responder)")
        elif int(comp) == 1:
            lbl_s1.config(text="-> Comparables")
        else:
            lbl_s1.config(text="-> NO comparables (Paso 2 no aplica)")
        # stage 2 enabled only if comparable == 1
        enable_s2 = (not pd.isna(comp)) and int(comp) == 1
        for b in s2_btns.winfo_children():
            b.config(state="normal" if enable_s2 else "disabled")
        if enable_s2 and not pd.isna(sc):
            lbl_s2.config(text=f"-> {int(sc):+d}")
        elif enable_s2:
            lbl_s2.config(text="(elige un nivel)")
        else:
            lbl_s2.config(text="")
        btn_prev.config(state="normal" if i > 0 else "disabled")
        lbl_saved.config(text="guardado" if not pd.isna(comp) else "")

    def go(delta):
        state["idx"] = max(0, min(len(df) - 1, state["idx"] + delta))
        render()

    def finish_check():
        if df["HUMAN_COMPARABLE"].notna().all():
            messagebox.showinfo(
                "Listo",
                "Terminaste de etiquetar todos los pares.\n\n"
                "Ahora corre:\n  python judge_human_gui.py kappa")

    def set_stage1(val):
        i = state["idx"]
        df.at[i, "HUMAN_COMPARABLE"] = int(val)
        if val == 0:
            df.at[i, "HUMAN_SCORE"] = np.nan
            save(); render()
            finish_check()
            root.after(120, lambda: go(+1))          # auto-advance
        else:
            save(); render()                          # wait for stage 2

    def set_stage2(val):
        i = state["idx"]
        if pd.isna(df.at[i, "HUMAN_COMPARABLE"]) or int(df.at[i, "HUMAN_COMPARABLE"]) != 1:
            df.at[i, "HUMAN_COMPARABLE"] = 1
        df.at[i, "HUMAN_SCORE"] = int(val)
        save(); render()
        finish_check()
        root.after(120, lambda: go(+1))               # auto-advance

    # stage-1 buttons
    ttk.Button(s1, text="Comparables  [C]", command=lambda: set_stage1(1)).pack(
        side="left", padx=(0, 8))
    ttk.Button(s1, text="No comparables  [N]", command=lambda: set_stage1(0)).pack(
        side="left")

    # stage-2 buttons
    for label, val in [("+2 Acuerdo fuerte  [1]", 2), ("+1 Acuerdo parcial  [2]", 1),
                       ("-1 Contra. parcial  [3]", -1), ("-2 Contra. directa  [4]", -2)]:
        ttk.Button(s2_btns, text=label,
                   command=lambda v=val: set_stage2(v)).pack(side="left", padx=(0, 6))

    btn_prev.config(command=lambda: go(-1))
    btn_next.config(command=lambda: go(+1))

    # keyboard shortcuts
    root.bind("c", lambda e: set_stage1(1)); root.bind("C", lambda e: set_stage1(1))
    root.bind("n", lambda e: set_stage1(0)); root.bind("N", lambda e: set_stage1(0))
    root.bind("1", lambda e: set_stage2(2))
    root.bind("2", lambda e: set_stage2(1))
    root.bind("3", lambda e: set_stage2(-1))
    root.bind("4", lambda e: set_stage2(-2))
    root.bind("<Left>", lambda e: go(-1))
    root.bind("<Right>", lambda e: go(+1))
    root.protocol("WM_DELETE_WINDOW", lambda: (save(), root.destroy()))

    render()
    root.mainloop()


# --------------------------------------------------------------- kappa
def _cohen(a, b, weights=None):
    from sklearn.metrics import cohen_kappa_score
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0:
        return np.nan
    labels = sorted(set(np.concatenate([a, b]).tolist()))
    try:
        return float(cohen_kappa_score(a, b, labels=labels, weights=weights))
    except Exception:
        return np.nan


def cmd_kappa(args):
    df = pd.read_csv(SHEET)
    lab = df[df["HUMAN_COMPARABLE"].notna()].copy()
    if len(lab) == 0:
        raise SystemExit("No labels yet. Open the GUI and annotate first.")

    res = {"n_labeled": int(len(lab)), "n_total": int(len(df))}

    # Stage 1: comparable? (binary)
    js1 = (lab["JUDGE_STAGE1"].fillna(0).astype(int) == 1).astype(int)
    hs1 = lab["HUMAN_COMPARABLE"].astype(int)
    res["stage1_kappa"] = _cohen(js1, hs1)
    res["stage1_pct_agree"] = float((js1.to_numpy() == hs1.to_numpy()).mean())
    res["stage1_n"] = int(len(lab))

    # Stage 2: only where BOTH called it comparable
    both = lab[(js1.to_numpy() == 1) & (hs1.to_numpy() == 1)].copy()
    both = both[both["HUMAN_SCORE"].notna() & both["JUDGE_FINAL"].notna()]
    if len(both):
        jf = both["JUDGE_FINAL"].astype(int)
        hf = both["HUMAN_SCORE"].astype(int)
        res["stage2_n"] = int(len(both))
        res["stage2_kappa_nominal"] = _cohen(jf, hf)
        res["stage2_kappa_quadratic"] = _cohen(jf, hf, weights="quadratic")
        res["stage2_pct_exact"] = float((jf.to_numpy() == hf.to_numpy()).mean())
        # collapsed agree(+)/disagree(-)
        js = np.sign(jf.to_numpy()); hsg = np.sign(hf.to_numpy())
        res["stage2_sign_kappa"] = _cohen(js, hsg)
        res["stage2_sign_pct_agree"] = float((js == hsg).mean())
    else:
        res["stage2_n"] = 0

    # per-block stage-1 agreement (quick view)
    per = {}
    for b, g in lab.groupby("BLOCK"):
        jj = (g["JUDGE_STAGE1"].fillna(0).astype(int) == 1).astype(int).to_numpy()
        hh = g["HUMAN_COMPARABLE"].astype(int).to_numpy()
        per[BLOCK_NAME.get(int(b), str(b))] = {
            "n": int(len(g)), "stage1_kappa": _cohen(jj, hh),
            "stage1_pct_agree": float((jj == hh).mean())}
    res["per_block_stage1"] = per

    with open(os.path.join(OUTDIR, "judge_human_kappa.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("=== judge-vs-human agreement ===")
    print(f"labeled {res['n_labeled']}/{res['n_total']} pairs")
    print(f"\nStage 1 (comparable?):  kappa = {res['stage1_kappa']:.3f}  "
          f"| exact = {res['stage1_pct_agree']:.1%}  (n={res['stage1_n']})")
    if res["stage2_n"]:
        print(f"Stage 2 (agreement, n={res['stage2_n']} both-comparable):")
        print(f"  nominal kappa    = {res['stage2_kappa_nominal']:.3f}")
        print(f"  quadratic kappa  = {res['stage2_kappa_quadratic']:.3f}  (ordinal)")
        print(f"  exact match      = {res['stage2_pct_exact']:.1%}")
        print(f"  agree/disagree kappa = {res['stage2_sign_kappa']:.3f}  "
              f"| pct = {res['stage2_sign_pct_agree']:.1%}")
    else:
        print("Stage 2: no pairs where both sides said 'comparable' yet.")
    print("\nwrote", os.path.join(OUTDIR, "judge_human_kappa.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    ms = sub.add_parser("make-sheet")
    ms.add_argument("--n-per-block", type=int, default=25)
    ms.add_argument("--seed", type=int, default=20260813)
    ms.add_argument("--overwrite", action="store_true")
    ms.set_defaults(func=cmd_make_sheet)
    sub.add_parser("gui").set_defaults(func=cmd_gui)
    sub.add_parser("kappa").set_defaults(func=cmd_kappa)
    args = ap.parse_args()
    if args.cmd is None:                    # default: launch the GUI
        args.func = cmd_gui
    args.func(args)


if __name__ == "__main__":
    main()
