"""#11 -- REMOTE (GPU) half of the judge run-to-run reliability check.

The ONLY file that needs to live on the GPU box. It reads a small pairs CSV
(produced locally by judge_reliability_analyze.py `sample`), re-scores every pair K
times with the EXACT judge prompts / sampling params from pipeline.stages.judge, and
writes a compact scores parquet to download back. No VLM is re-run and no local data
is needed here beyond the pairs CSV -- the repo (on GitHub) supplies the judge code.

Place this file at the repo root (or anywhere the `pipeline` package is importable)
on the remote, then:

  python judge_reliability_generate.py \
      --pairs-csv judge_reliability_pairs.csv \
      --model Qwen/Qwen3-4B --base-url http://localhost:8000/v1 \
      --runs 5 --temperature 1.0 --out judge_reliability_runs_Qwen_Qwen3-4B.parquet

Then `scp` the output parquet back and run:
  python judge_reliability_analyze.py report --runs-parquet <that file>

Independent second judge: point --model / --base-url at e.g. a Mistral model at
--temperature 0 and save to a different --out.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import numpy as np
import pandas as pd
import httpx

# repo root must be importable so we reuse the identical judge code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pipeline.stages.judge import (
        judge_row_answers_async, extract_json_dict,
        JUDGE_TEMPLATE_STAGE1, JUDGE_TEMPLATE_STAGE2,
    )
except ImportError as e:
    sys.exit(f"Cannot import pipeline.stages.judge ({e}). "
             f"Run this from the repo root on the remote.")


async def _score_stage(row, client, model, temperature, max_tokens, system_prompt,
                       max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            payload = await judge_row_answers_async(
                row=row, client=client, model=model, temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt)
            return extract_json_dict(payload["raw_output"] or "").get("Score", np.nan)
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(2.0 * (2 ** (attempt - 1)))
    return np.nan


async def _score_pair(row, client, model, temperature, max_tokens, sem):
    async with sem:
        s1 = await _score_stage(row, client, model, temperature, max_tokens,
                                JUDGE_TEMPLATE_STAGE1)
        s2 = np.nan
        if s1 == 1:
            s2 = await _score_stage(row, client, model, temperature, max_tokens,
                                    JUDGE_TEMPLATE_STAGE2)
    return s1, s2, (s2 if s1 == 1 else np.nan)


async def run_all(pairs, args):
    headers = {"Authorization": f"Bearer {args.api_key}"}
    records = []
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers,
                                 timeout=args.timeout) as client:
        sem = asyncio.Semaphore(args.concurrency)
        for run_id in range(args.runs):
            tasks = [_score_pair(r, client, args.model, args.temperature,
                                 args.max_tokens, sem)
                     for _, r in pairs.iterrows()]
            results = await asyncio.gather(*tasks)
            for (idx, r), (s1, s2, fin) in zip(pairs.iterrows(), results):
                records.append(dict(pair=int(idx), run=run_id, BLOCK=int(r["BLOCK"]),
                                    STAGE1=s1, STAGE2=s2, FINAL=fin))
            print(f"  run {run_id + 1}/{args.runs} done", flush=True)
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", default="judge_reliability_pairs.csv")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs_csv)
    print(f"{len(pairs)} pairs, blocks {pairs.BLOCK.value_counts().to_dict()}; "
          f"{args.runs} runs at T={args.temperature} on {args.model}", flush=True)
    runs = asyncio.run(run_all(pairs, args))
    out = args.out or ("judge_reliability_runs_"
                       + "".join(c if c.isalnum() else "_" for c in args.model)
                       + ".parquet")
    runs.to_parquet(out)
    print(f"wrote {out}  ({len(runs)} rows) -- scp this back to the local box")


if __name__ == "__main__":
    main()
