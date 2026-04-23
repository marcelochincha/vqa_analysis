from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    workspace_root: Path
    data_file: Path = Path("data/r2_cleaned.csv")
    embeddings_file: Path = Path("outputs/output_embeddings_cleaned_allmpnet/embeddings_cache_keyed_allmpnet.pkl")
    outdir: Path = Path("outputs/pipeline")

    def resolve(self, relative: Path) -> Path:
        return self.workspace_root / relative

    def out_path(self, stage: str, filename: str = "") -> Path:
        path = self.outdir / stage
        if filename:
            path = path / filename
        return self.resolve(path)


def get_config(
    workspace_root: Path | None = None,
    data_file: Path | None = None,
    embeddings_file: Path | None = None,
    outdir: Path | None = None,
) -> PipelineConfig:
    root = workspace_root or Path(__file__).resolve().parents[1]
    return PipelineConfig(
        workspace_root=root,
        data_file=data_file or root / "data/r2_cleaned.csv",
        embeddings_file=embeddings_file or root / "external_embeds/allmpnet_batch1_r2_embeddings_cache_keyed.pkl",
        outdir=outdir or root / "outputs/pipeline",
    )