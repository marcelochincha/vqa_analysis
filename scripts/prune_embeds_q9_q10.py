"""Copy an embeddings pickle while dropping Block-2 Q9/Q10 entries.

After running this, point generate_embeddings.py at the binarized CSV with
--resume and the new output path: only the 8600 Q9/Q10 keys will be missing
and get re-encoded with their new "True"/"False" text, while every other
embedding is reused from the original pickle.

Usage:
    python scripts/prune_embeds_q9_q10.py \
        --input  external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl \
        --output external_embeds/allmpnet_batch1_r2_q9q10binary_cache_keyed.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

TARGET_QUESTIONS = {9, 10}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("external_embeds/allmpnet_batch1_r2_q9q10binary_cache_keyed.pkl"),
    )
    args = parser.parse_args()

    with args.input.open("rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache):,} entries from {args.input}")

    # Key layout: (agent, video, question_num, repetition)
    pruned = {k: v for k, v in cache.items() if k[2] not in TARGET_QUESTIONS}
    dropped = len(cache) - len(pruned)
    print(f"Dropped {dropped:,} entries for Q{sorted(TARGET_QUESTIONS)}")
    print(f"Remaining: {len(pruned):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(pruned, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
