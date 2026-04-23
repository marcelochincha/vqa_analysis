from __future__ import annotations

from pipeline.stages.bias import run as bias_run
from pipeline.stages.cosine import run as cosine_run
from pipeline.stages.embed import run as embed_run
from pipeline.stages.judge import run as judge_run
from pipeline.stages.preprocess import run as preprocess_run
from pipeline.stages.rsa import run as rsa_run

STAGE_RUNNERS = {
    "preprocess": preprocess_run,
    "embed": embed_run,
    "cosine": cosine_run,
    "rsa": rsa_run,
    "bias": bias_run,
    #"judge": judge_run,
}

__all__ = ["STAGE_RUNNERS"]