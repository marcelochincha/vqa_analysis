from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --- Heatmap tri-point blended inner-grid (A -> white -> B) -------------------
# Single source of truth for the "blender box" drawn by pipeline.style.styled_heatmap
# on every heatmap. The band replaces the fixed white separator lines and renders
# identically in PNG and SVG/PDF.
HEATMAP_BLEND_ENABLED = True   # set False to fall back to fixed white grid lines
HEATMAP_BLEND_WIDTH = 0.05     # band width as a fraction of one cell
HEATMAP_BLEND_RES = 24         # raster px per cell for the overlay (higher = smoother)
HEATMAP_BLEND_ALPHA = 0.8      # overlay opacity


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