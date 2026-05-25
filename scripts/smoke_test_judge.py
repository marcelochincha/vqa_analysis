"""Smoke test for the judge stage.

Builds a tiny synthetic subset (a handful of agent pairs), runs judge.run()
against your local vLLM, then inspects the resulting parquet to confirm:
  * Stage 1 scores were populated
  * Stage 2 scores were populated for comparable pairs
  * The model's reasoning_content (the <think> block) was captured

Usage:
    python scripts/smoke_test_judge.py
    python scripts/smoke_test_judge.py --base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.stages import judge


GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def build_tiny_subset(source_csv: Path, out_csv: Path) -> int:
    df = pd.read_csv(source_csv, keep_default_na=False)
    df = df[df["REPETITION"] == 1]
    df = df[df["BLOCK"] == 1]

    videos = sorted(df["VIDEO"].unique())[:1]
    df = df[df["VIDEO"].isin(videos)]
    df = df[df["QUESTION_NUM"] == 1]

    picked = []
    for prefix in ("human_lima_1", "human_nyc_1"):
        match = [a for a in df["AGENT"].unique() if a.lower().startswith(prefix)]
        if match:
            picked.append(match[0])
    vlms = [a for a in df["AGENT"].unique() if "human" not in a.lower()]
    if vlms:
        picked.append(sorted(vlms)[0])

    df = df[df["AGENT"].isin(picked)]
    df.to_csv(out_csv, index=False)
    return len(df)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
    suffix = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/r2_cleaned.csv"))
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"{RED}ERROR{RESET}: {args.data} not found. Run preprocess first.")
        return 1

    with tempfile.TemporaryDirectory(prefix="judge_smoke_") as tmp:
        tmp_dir = Path(tmp)
        tiny_csv = tmp_dir / "tiny.csv"
        n = build_tiny_subset(args.data, tiny_csv)
        print(f"{BOLD}[setup]{RESET} tiny dataset = {n} rows -> {tiny_csv}")

        config = PipelineConfig(
            workspace_root=Path.cwd(),
            data_file=tiny_csv.resolve(),
            outdir=(tmp_dir / "outputs").resolve(),
        )

        print(f"{BOLD}[run]{RESET}  calling judge.run() against {args.base_url} ...")
        try:
            judge.run(
                config,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=1.0,
                max_tokens=args.max_tokens,
                concurrency=4,
                batch_size=8,
                checkpoint_every_batches=1,
            )
        except Exception as e:
            print(f"  {RED}judge.run() raised: {type(e).__name__}: {e}{RESET}")
            print(f"  {DIM}(checking parquet anyway in case plotting was the problem){RESET}")

        parquet = tmp_dir / "outputs" / "judge" / "llm_agreement_scores.parquet"
        if not parquet.exists():
            print(f"\n{RED}FATAL{RESET}: parquet was never written at {parquet}")
            return 1

        df = pd.read_parquet(parquet)
        print(f"\n{BOLD}[verify]{RESET} loaded parquet -> {len(df)} pairs, {len(df.columns)} columns")
        print(f"          columns: {list(df.columns)}\n")

        ok = True
        ok &= check("Parquet has at least 1 pair", len(df) > 0)
        for col in ("STAGE1_SCORE", "STAGE1_EVALUATION", "STAGE1_REASONING",
                    "STAGE2_SCORE", "STAGE2_EVALUATION", "STAGE2_REASONING",
                    "FINAL_SCORE", "QUESTION"):
            ok &= check(f"Column {col} exists", col in df.columns)

        stage1_filled = int(df["STAGE1_SCORE"].notna().sum())
        ok &= check(
            "Every pair has STAGE1_SCORE",
            stage1_filled == len(df),
            f"{stage1_filled}/{len(df)} filled",
        )

        scores1 = df["STAGE1_SCORE"].dropna().unique().tolist()
        ok &= check(
            "STAGE1_SCORE values in {0, 1}",
            set(scores1).issubset({0.0, 1.0}),
            f"saw {scores1}",
        )

        reasoning1_len = df["STAGE1_REASONING"].astype(str).str.len()
        ok &= check(
            "STAGE1_REASONING is non-empty (model emitted <think>)",
            (reasoning1_len > 20).all(),
            f"min={int(reasoning1_len.min())} median={int(reasoning1_len.median())} max={int(reasoning1_len.max())} chars",
        )

        eval1_len = df["STAGE1_EVALUATION"].astype(str).str.len()
        ok &= check(
            "STAGE1_EVALUATION is non-empty",
            (eval1_len > 10).all(),
            f"median={int(eval1_len.median())} chars",
        )

        comparable = df[df["STAGE1_SCORE"] == 1]
        if len(comparable):
            stage2_filled = int(comparable["STAGE2_SCORE"].notna().sum())
            ok &= check(
                "Every comparable pair has STAGE2_SCORE",
                stage2_filled == len(comparable),
                f"{stage2_filled}/{len(comparable)} filled",
            )
            scores2 = comparable["STAGE2_SCORE"].dropna().unique().tolist()
            ok &= check(
                "STAGE2_SCORE values in {-2, -1, 1, 2}",
                set(scores2).issubset({-2.0, -1.0, 1.0, 2.0}),
                f"saw {scores2}",
            )
            reasoning2_len = comparable["STAGE2_REASONING"].astype(str).str.len()
            ok &= check(
                "STAGE2_REASONING is non-empty",
                (reasoning2_len > 20).all(),
                f"median={int(reasoning2_len.median())} chars",
            )
            ok &= check(
                "FINAL_SCORE = STAGE2_SCORE for comparable pairs",
                (comparable["FINAL_SCORE"] == comparable["STAGE2_SCORE"]).all(),
            )
        else:
            print(f"  {DIM}(no comparable pairs in this tiny sample, skipping Stage 2 checks){RESET}")

        print(f"\n{BOLD}========== SAMPLE OUTPUT (one row, full content) =========={RESET}")
        row = df.iloc[0]
        for col in ("AGENT_I", "AGENT_J", "QUESTION", "ANSWER_I", "ANSWER_J",
                    "STAGE1_SCORE", "STAGE1_EVALUATION", "STAGE1_REASONING",
                    "STAGE2_SCORE", "STAGE2_EVALUATION", "STAGE2_REASONING",
                    "FINAL_SCORE"):
            if col not in df.columns:
                continue
            val = str(row[col])
            if len(val) > 500:
                val = val[:500] + f"  ...(+{len(val) - 500} chars)"
            print(f"\n{BOLD}{col}{RESET}: {val}")
        print(f"\n{BOLD}========================================================={RESET}\n")

        if ok:
            print(f"{GREEN}{BOLD}ALL SMOKE CHECKS PASSED{RESET}")
            return 0
        print(f"{RED}{BOLD}SMOKE CHECKS FAILED{RESET}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
