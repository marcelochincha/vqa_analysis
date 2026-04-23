from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def load_embeddings_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def resolve_path(base: Path, relative: Path | str | None) -> Path:
    if relative is None:
        return base
    rel = Path(relative)
    return rel if rel.is_absolute() else base / rel