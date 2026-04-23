from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from openai import AsyncOpenAI
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.style import apply_style
from pipeline.utils.io import load_csv


JUDGE_TEMPLATE = """
You are a strict and impartial evaluator.

Task:
Assess semantic agreement between Response A and Response B with respect to the same question about an urban driving scene.

Rules:
- Compare only explicit factual claims.
- Ignore style, tone, verbosity, and grammar.
- Do not reward verbosity.
- Do not infer or assume unstated facts.
- If one response is vague/empty and the other is specific, score 0 unless there is explicit contradiction.
- Use the full scale only when justified by explicit claim overlap/conflict.

Scoring:
+2 = Strong agreement: same core claims, no meaningful differences.
+1 = Partial agreement: mostly aligned, minor differences or omissions.
0 = No clear relation: little/no overlapping claims, or insufficient comparable content.
-1 = Partial contradiction: at least one important claim conflicts.
-2 = Direct contradiction: opposing claims about the same fact.

Output:
Return ONLY valid JSON (no markdown, no extra text):
{{
  "Evaluation": "<1-4 concise sentences based only on explicit claims>",
  "Score": -2 | -1 | 0 | 1 | 2
}}

Input:
    Response A: {res_a}
    Response B: {res_b}
"""


def normalize_message_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in value).strip()
    return str(value).strip()


def extract_json_dict(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in model output")
        return json.loads(match.group(0))


async def judge_row_answers_async(row, client: AsyncOpenAI, model: str) -> dict:
    prompt = JUDGE_TEMPLATE.format(res_a=row["ANSWER_I"], res_b=row["ANSWER_J"])
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an impartial judge."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    msg = completion.choices[0].message
    return {
        "raw_output": normalize_message_field(msg.content),
        "reasoning_content": normalize_message_field(getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)),
    }


async def compare_one(row, sem, client, model):
    async with sem:
        try:
            result_payload = await judge_row_answers_async(row, client, model)
            result = extract_json_dict(result_payload["raw_output"] or "")
            score = result.get("Score", np.nan)
            evaluation = result.get("Evaluation", "")
            reasoning = result_payload.get("reasoning_content", "")
        except Exception:
            score = np.nan
            evaluation = ""
            reasoning = ""
    return score, evaluation, reasoning


async def score_dataframe_async(df: pd.DataFrame, checkpoint_path: str, client: AsyncOpenAI, model: str, concurrency: int = 16, batch_size: int = 128):
    sem = asyncio.Semaphore(concurrency)
    for start in tqdm(range(0, len(df), batch_size), desc="Scoring batches"):
        end = min(start + batch_size, len(df))
        rows = [df.iloc[i] for i in range(start, end)]
        results = await asyncio.gather(*(compare_one(row, sem, client, model) for row in rows))
        for i, (score, evaluation, reasoning) in zip(range(start, end), results):
            df.at[i, "SCORE"] = score
            df.at[i, "EVALUATION"] = evaluation
            df.at[i, "REASONING_CONTENT"] = reasoning
        df.to_parquet(checkpoint_path)
    return df


def plot_judge(rsa_df: pd.DataFrame, agents, df_answers: pd.DataFrame, out_path: Path) -> Path:
    apply_style()
    nrows = 2
    ncols = 4
    fig, ax = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for id_r, region in enumerate(df_answers["VIDEO_SECTOR"].unique()[::-1]):
        for block in df_answers["BLOCK"].astype(int).unique():
            c_ax = ax[id_r, block - 1]
            df_block = rsa_df[(rsa_df["VIDEO_SECTOR"] == region) & (rsa_df["BLOCK"] == block)]
            rsa_matrix = df_block.pivot(index="AGENT_I", columns="AGENT_J", values="SCORE")
            rsa_matrix = rsa_matrix.reindex(index=agents, columns=agents)
            sns.heatmap(rsa_matrix.to_numpy(), annot=False, xticklabels=rsa_matrix.columns, yticklabels=rsa_matrix.columns, cmap=cmap, ax=c_ax, square=True, vmin=-2, vmax=2)
            c_ax.set_title(f"Region: {region}, Block: {block}", fontsize=16, weight="bold")

    fig.suptitle("LLM Judge - Scores by Agent and Block", fontsize=24, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run(config: PipelineConfig, model: str = "Qwen/Qwen3-4B", base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY", concurrency: int = 16, batch_size: int = 128) -> Path:
    data_path = config.resolve(config.data_file)
    outdir = config.out_path("judge")
    checkpoint_path = outdir / "llm_agreement_scores.parquet"
    outdir.mkdir(parents=True, exist_ok=True)

    df_answers = load_csv(data_path, keep_default_na=False)
    df_answers = df_answers[df_answers["REPETITION"] == 1].reset_index(drop=True)
    df_answers = df_answers[df_answers["BLOCK"] != 2].reset_index(drop=True)
    df_answers["VIDEO_SECTOR"] = df_answers["VIDEO"].apply(lambda x: "Lima" if int(x.split("_")[1]) <= 100 else "NYC")

    keys = ["VIDEO", "QUESTION_NUM", "VIDEO_SECTOR", "BLOCK", "AGENT", "ANSWER"]
    df_comp = pd.merge(df_answers[keys], df_answers[keys], on=["VIDEO", "QUESTION_NUM", "VIDEO_SECTOR", "BLOCK"], suffixes=("_I", "_J"))

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    df_comp["SCORE"] = np.nan
    df_comp["EVALUATION"] = ""
    df_comp["REASONING_CONTENT"] = ""
    df_comp = asyncio.run(score_dataframe_async(df_comp, str(checkpoint_path), client, model, concurrency=concurrency, batch_size=batch_size))

    agg_df = df_comp.groupby(["VIDEO", "QUESTION_NUM", "BLOCK", "VIDEO_SECTOR", "AGENT_I", "AGENT_J"], as_index=False)["SCORE"].mean()
    agg_df2 = agg_df.groupby(["AGENT_I", "AGENT_J", "BLOCK", "VIDEO_SECTOR"], as_index=False)["SCORE"].mean()

    out_path = outdir / "judge_scores.png"
    plot_judge(agg_df2, agents=df_answers["AGENT"].unique(), df_answers=df_answers, out_path=out_path)
    return out_path