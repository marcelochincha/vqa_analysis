from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


STAGE_CACHE_FILES: dict[str, list[str]] = {
    "cosine": ["cosine_similarity_data.parquet"],
    "rsa": ["rsa_correlations.parquet"],
    "embed": ["pca_coords.parquet"],
    "judge": [
        "llm_agreement_scores.parquet",
        "llm_agreement_scores.parquet.tmp",
        "comparable_pairs.parquet",
        "non_comparable_pairs.parquet",
    ],
}


def save_dataframe(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    os.replace(tmp_path, path)
    return path


def load_dataframe(path: Path | str) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def cached_dataframe(
    path: Path | str,
    compute_fn: Callable[[], pd.DataFrame],
    *,
    force: bool = False,
    label: str = "",
) -> pd.DataFrame:
    path = Path(path)
    tag = f"[cache:{label}]" if label else "[cache]"

    if not force:
        cached = load_dataframe(path)
        if cached is not None:
            print(f"{tag} Loaded {len(cached)} rows from {path}")
            return cached

    print(f"{tag} Computing fresh -> will save to {path}")
    df = compute_fn()
    save_dataframe(df, path)
    print(f"{tag} Saved {len(df)} rows to {path}")
    return df


def clear_cache(path: Path | str) -> bool:
    path = Path(path)
    if path.exists():
        path.unlink()
        return True
    return False


def clear_stage_caches(config, stages: Optional[list[str]] = None) -> dict[str, list[Path]]:
    targets = stages if stages else list(STAGE_CACHE_FILES.keys())
    removed: dict[str, list[Path]] = {}
    for stage in targets:
        files = STAGE_CACHE_FILES.get(stage, [])
        removed_files: list[Path] = []
        for fname in files:
            full = config.out_path(stage, fname)
            if clear_cache(full):
                removed_files.append(full)
                print(f"[cache:{stage}] removed {full}")
        removed[stage] = removed_files
    return removed
