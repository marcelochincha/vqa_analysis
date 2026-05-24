from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import get_config
from pipeline.stages import STAGE_RUNNERS


def main():
    parser = argparse.ArgumentParser(prog="pipeline", description="VQA Analysis Pipeline")
    parser.add_argument("stages", nargs="*", help="Stages to run (preprocess, embed, cosine, rsa, bias, judge)")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--list", action="store_true", help="List available stages")
    parser.add_argument("--progress", action="store_true", help="Show progress bars for long computations")
    parser.add_argument("--data", type=Path, help="Input CSV data file")
    parser.add_argument("--embeddings", type=Path, help="Embeddings cache file")
    parser.add_argument("--outdir", type=Path, help="Output directory")
    parser.add_argument("--human-csv", type=Path, help="Raw human answers CSV file")
    parser.add_argument("--vlm-dir", type=Path, help="Directory containing VLM JSON files")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="LLM model for judge")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--temperature", type=float, default=0.6, help="Judge sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max tokens for judge responses")
    parser.add_argument("--concurrency", type=int, default=16, help="Judge concurrent requests")
    parser.add_argument("--batch-size", type=int, default=128, help="Judge batch size")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Judge checkpoint cadence in batches")
    args = parser.parse_args()

    if args.list:
        print("Available stages:")
        for name in STAGE_RUNNERS:
            print(f"  {name}")
        return 0

    stages = list(args.stages) if args.stages else []
    if args.all:
        stages = list(STAGE_RUNNERS.keys())

    if not stages:
        parser.print_help()
        return 1

    config = get_config(
        data_file=args.data,
        embeddings_file=args.embeddings,
        outdir=args.outdir,
    )

    results = {}
    for name in stages:
        if name not in STAGE_RUNNERS:
            print(f"Unknown stage: {name}", file=sys.stderr)
            continue
        print(f"Running stage: {name}...")
        try:
            runner = STAGE_RUNNERS[name]
            if name == "judge":
                result = runner(
                    config,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    concurrency=args.concurrency,
                    batch_size=args.batch_size,
                    checkpoint_every_batches=args.checkpoint_every,
                )
            elif name == "preprocess":
                result = runner(config, human_csv=args.human_csv, vlm_dir=args.vlm_dir)
            elif name in ("cosine", "rsa"):
                result = runner(config, show_progress=args.progress)
            else:
                result = runner(config)
            results[name] = result
            print(f"  -> {result}")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue

    if results:
        print(f"\nCompleted: {', '.join(results.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())