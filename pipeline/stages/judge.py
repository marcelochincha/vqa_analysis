from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.style import apply_style
from pipeline.utils.io import load_csv
from pipeline.utils.metrics import get_video_sector


# =========================================================
# STAGE 1 PROMPT
# =========================================================

JUDGE_TEMPLATE_STAGE1 = """
You are an expert logical analyst.

Task:
Determine if Response A and Response B provide claims that can be compared for agreement or contradiction regarding the [Question].

Rules:
- Mark as 1 (COMPARABLE) if:
    1. Both responses provide a factual answer to the [Question].
    2. They describe the same object's state.
    3. The responses are "on the same page" even if they disagree.

- Mark as 0 (NOT_COMPARABLE) if:
    1. The responses talk past each other.
    2. One response provides facts while the other says "I don't know".
    3. They discuss different objects entirely.

Output ONLY valid JSON:
{
  "Evaluation": "Brief reasoning",
  "Score": 1 | 0
}
"""


# =========================================================
# STAGE 2 PROMPT
# =========================================================

JUDGE_TEMPLATE_STAGE2 = """
You are a strict and impartial evaluator of factual alignment.

Task:
Compare the semantic agreement between Response A and Response B regarding the [Question].

Scoring Scale:
+2 Strong Agreement
+1 Partial Agreement
-1 Partial Contradiction
-2 Direct Contradiction

Output ONLY valid JSON:
{
  "Evaluation": "Brief reasoning",
  "Score": 2 | 1 | -1 | -2
}
"""


# =========================================================
# USER TEMPLATE
# =========================================================

USER_TEMPLATE = """
Input:
[Question]: {question}

[Response A]:
{res_a}

[Response B]:
{res_b}
"""


# =========================================================
# HELPERS
# =========================================================

def normalize_message_field(value) -> str:

    if value is None:
        return ""

    if isinstance(value, list):

        return "".join(
            part.get("text", "")
            if isinstance(part, dict)
            else str(part)
            for part in value
        ).strip()

    return str(value).strip()


def extract_json_dict(text: str) -> dict:

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip()

        text = re.sub(
            r"```$",
            "",
            text.strip(),
        ).strip()

    try:
        return json.loads(text)

    except Exception:

        match = re.search(
            r"\{[\s\S]*\}",
            text,
        )

        if not match:
            raise ValueError(
                "No JSON object found"
            )

        return json.loads(
            match.group(0)
        )


# =========================================================
# SINGLE MODEL CALL
# =========================================================

async def judge_row_answers_async(
    row,
    client: httpx.AsyncClient,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
):

    prompt = USER_TEMPLATE.format(
        question=row["QUESTION"],
        res_a=row["ANSWER_I"],
        res_b=row["ANSWER_J"],
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = await client.post(
        "/chat/completions",
        json=payload,
    )

    response.raise_for_status()

    data = response.json()

    msg = data["choices"][0]["message"]

    return {
        "raw_output": normalize_message_field(
            msg.get("content")
        ),
        "reasoning_content": normalize_message_field(
            msg.get("reasoning_content")
            or msg.get("reasoning")
        ),
    }


# =========================================================
# RUN ONE STAGE
# =========================================================

async def run_stage(
    row,
    sem,
    client,
    model,
    temperature,
    max_tokens,
    system_prompt,
):

    async with sem:

        try:

            result_payload = await judge_row_answers_async(
                row=row,
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

            result = extract_json_dict(
                result_payload["raw_output"] or ""
            )

            score = result.get(
                "Score",
                np.nan,
            )

            evaluation = result.get(
                "Evaluation",
                "",
            )

            reasoning = result_payload.get(
                "reasoning_content",
                "",
            )

        except Exception as e:

            print(f"ERROR: {e}")

            score = np.nan
            evaluation = ""
            reasoning = ""

    return (
        score,
        evaluation,
        reasoning,
    )


# =========================================================
# MAIN SCORING LOOP
# =========================================================

async def score_dataframe_async(
    df: pd.DataFrame,
    checkpoint_path: str,
    client: httpx.AsyncClient,
    model: str,
    temperature: float,
    max_tokens: int,
    concurrency: int = 16,
    batch_size: int = 128,
    checkpoint_every_batches: int = 10,
):

    sem = asyncio.Semaphore(concurrency)

    # =====================================================
    # RESUME ONLY MISSING ROWS
    # =====================================================

    pending_indices = df[
        (
            df["STAGE1_SCORE"].isna()
        )
        |
        (
            (df["STAGE1_SCORE"] == 1)
            &
            (df["STAGE2_SCORE"].isna())
        )
    ].index.tolist()

    print(
        f"Pending rows: {len(pending_indices)}"
    )

    for batch_i, start_idx in enumerate(

        tqdm(
            range(
                0,
                len(pending_indices),
                batch_size,
            ),
            desc="Scoring batches",
        ),

        start=1,
    ):

        batch_indices = pending_indices[
            start_idx:start_idx + batch_size
        ]

        rows = [
            df.loc[i]
            for i in batch_indices
        ]

        # =================================================
        # STAGE 1
        # =================================================

        rows_need_stage1 = []
        rows_need_stage1_idx = []

        for idx, row in zip(
            batch_indices,
            rows,
        ):

            if pd.isna(
                df.at[idx, "STAGE1_SCORE"]
            ):

                rows_need_stage1.append(row)
                rows_need_stage1_idx.append(idx)

        if rows_need_stage1:

            stage1_results = await asyncio.gather(
                *(
                    run_stage(
                        row=row,
                        sem=sem,
                        client=client,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=JUDGE_TEMPLATE_STAGE1,
                    )
                    for row in rows_need_stage1
                )
            )

            for global_idx, (
                score,
                evaluation,
                reasoning,
            ) in zip(
                rows_need_stage1_idx,
                stage1_results,
            ):

                df.at[
                    global_idx,
                    "STAGE1_SCORE",
                ] = score

                df.at[
                    global_idx,
                    "STAGE1_EVALUATION",
                ] = evaluation

                df.at[
                    global_idx,
                    "STAGE1_REASONING",
                ] = reasoning

        # =================================================
        # STAGE 2
        # =================================================

        comparable_rows = []

        for idx, row in zip(
            batch_indices,
            rows,
        ):

            if (
                df.at[idx, "STAGE1_SCORE"] == 1
                and
                pd.isna(
                    df.at[idx, "STAGE2_SCORE"]
                )
            ):

                comparable_rows.append(
                    (
                        idx,
                        row,
                    )
                )

        if comparable_rows:

            stage2_results = await asyncio.gather(
                *(
                    run_stage(
                        row=row,
                        sem=sem,
                        client=client,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_prompt=JUDGE_TEMPLATE_STAGE2,
                    )
                    for _, row in comparable_rows
                )
            )

            for (
                global_idx,
                _,
            ), (
                score,
                evaluation,
                reasoning,
            ) in zip(
                comparable_rows,
                stage2_results,
            ):

                df.at[
                    global_idx,
                    "STAGE2_SCORE",
                ] = score

                df.at[
                    global_idx,
                    "STAGE2_EVALUATION",
                ] = evaluation

                df.at[
                    global_idx,
                    "STAGE2_REASONING",
                ] = reasoning

                df.at[
                    global_idx,
                    "FINAL_SCORE",
                ] = score

        # =================================================
        # NON COMPARABLE
        # =================================================

        non_comparable_indices = [
            idx
            for idx in batch_indices
            if df.at[idx, "STAGE1_SCORE"] != 1
        ]

        for idx in non_comparable_indices:

            df.at[
                idx,
                "FINAL_SCORE",
            ] = np.nan

        # =================================================
        # CHECKPOINT
        # =================================================

        if (
            checkpoint_every_batches > 0
            and batch_i % checkpoint_every_batches == 0
        ):

            print(
                f"Saving checkpoint batch {batch_i}"
            )

            df.to_parquet(
                checkpoint_path
            )

    # =====================================================
    # FINAL SAVE
    # =====================================================

    df.to_parquet(
        checkpoint_path
    )

    return df


# =========================================================
# PLOT
# =========================================================

def plot_judge(
    rsa_df: pd.DataFrame,
    agents,
    df_answers: pd.DataFrame,
    out_path: Path,
) -> Path:

    apply_style()

    nrows = 2
    ncols = 4

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5 * nrows),
        sharex=True,
        sharey=True,
    )

    cmap = sns.diverging_palette(
        220,
        20,
        as_cmap=True,
    )

    for id_r, region in enumerate(
        df_answers["VIDEO_SECTOR"].unique()[::-1]
    ):

        for block in df_answers[
            "BLOCK"
        ].astype(int).unique():

            c_ax = ax[id_r, block - 1]

            df_block = rsa_df[
                (rsa_df["VIDEO_SECTOR"] == region)
                &
                (rsa_df["BLOCK"] == block)
            ]

            rsa_matrix = df_block.pivot(
                index="AGENT_I",
                columns="AGENT_J",
                values="FINAL_SCORE",
            )

            rsa_matrix = rsa_matrix.reindex(
                index=agents,
                columns=agents,
            )

            sns.heatmap(
                rsa_matrix.to_numpy(),
                annot=False,
                xticklabels=rsa_matrix.columns,
                yticklabels=rsa_matrix.columns,
                cmap=cmap,
                ax=c_ax,
                square=True,
                vmin=-2,
                vmax=2,
            )

            c_ax.set_title(
                f"Region: {region}, Block: {block}",
                fontsize=16,
                weight="bold",
            )

    fig.suptitle(
        "LLM Judge - Scores by Agent and Block",
        fontsize=24,
        weight="bold",
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out_path


# =========================================================
# RUN
# =========================================================

def run(
    config: PipelineConfig,
    model: str = "Qwen/Qwen3-4B",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    temperature: float = 0.6,
    max_tokens: int = 32768,
    concurrency: int = 16,
    batch_size: int = 128,
    checkpoint_every_batches: int = 10,
) -> Path:

    data_path = config.resolve(
        config.data_file
    )

    outdir = config.out_path(
        "judge"
    )

    checkpoint_path = (
        outdir
        / "llm_agreement_scores.parquet"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # LOAD ANSWERS
    # =====================================================

    df_answers = load_csv(
        data_path,
        keep_default_na=False,
    )

    df_answers = df_answers[
        df_answers["REPETITION"] == 1
    ].reset_index(drop=True)

    df_answers = df_answers[
        df_answers["BLOCK"] != 2
    ].reset_index(drop=True)

    df_answers["VIDEO_SECTOR"] = (
        df_answers["VIDEO_SECTOR"].apply(
            get_video_sector
        )
    )

    # =====================================================
    # LOAD OR BUILD CHECKPOINT
    # =====================================================

    if checkpoint_path.exists():

        print(
            f"Loading checkpoint: {checkpoint_path}"
        )

        df_comp = pd.read_parquet(
            checkpoint_path
        )

    else:

        # =================================================
        # BUILD PAIRS
        # =================================================

        keys = [
            "VIDEO",
            "QUESTION_NUM",
            "QUESTION",
            "VIDEO_SECTOR",
            "BLOCK",
            "AGENT",
            "ANSWER",
        ]

        df_comp = pd.merge(
            df_answers[keys],
            df_answers[keys],
            on=[
                "VIDEO",
                "QUESTION_NUM",
                "VIDEO_SECTOR",
                "BLOCK",
            ],
            suffixes=(
                "_I",
                "_J",
            ),
        )

        # =================================================
        # REMOVE SELF COMPARISONS
        # =================================================

        df_comp = df_comp[
            df_comp["AGENT_I"]
            != df_comp["AGENT_J"]
        ].reset_index(drop=True)

        # =================================================
        # REMOVE DUPLICATES
        # =================================================

        df_comp["PAIR_KEY"] = df_comp.apply(
            lambda r: tuple(
                sorted(
                    [
                        r["AGENT_I"],
                        r["AGENT_J"],
                    ]
                )
            ),
            axis=1,
        )

        df_comp = df_comp.drop_duplicates(
            subset=[
                "VIDEO",
                "QUESTION_NUM",
                "BLOCK",
                "PAIR_KEY",
            ]
        ).reset_index(drop=True)

        # =================================================
        # INIT COLUMNS
        # =================================================

        df_comp["STAGE1_SCORE"] = np.nan
        df_comp["STAGE1_EVALUATION"] = ""
        df_comp["STAGE1_REASONING"] = ""

        df_comp["STAGE2_SCORE"] = np.nan
        df_comp["STAGE2_EVALUATION"] = ""
        df_comp["STAGE2_REASONING"] = ""

        df_comp["FINAL_SCORE"] = np.nan

    # =====================================================
    # RUN SCORING
    # =====================================================

    headers = (
        {"Authorization": f"Bearer {api_key}"}
        if api_key
        else None
    )

    client = httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=None,
    )

    try:

        df_comp = asyncio.run(
            score_dataframe_async(
                df_comp,
                str(checkpoint_path),
                client,
                model,
                temperature,
                max_tokens,
                concurrency=concurrency,
                batch_size=batch_size,
                checkpoint_every_batches=checkpoint_every_batches,
            )
        )

    finally:

        asyncio.run(
            client.aclose()
        )

    # =====================================================
    # SPLIT DATAFRAMES
    # =====================================================

    df_comp["IS_COMPARABLE"] = (
        df_comp["STAGE1_SCORE"] == 1
    )

    comparable_df = df_comp[
        df_comp["IS_COMPARABLE"]
    ].copy()

    non_comparable_df = df_comp[
        ~df_comp["IS_COMPARABLE"]
    ].copy()

    # =====================================================
    # SAVE DATA
    # =====================================================

    comparable_df.to_parquet(
        outdir / "comparable_pairs.parquet"
    )

    non_comparable_df.to_parquet(
        outdir / "non_comparable_pairs.parquet"
    )

    comparable_df.to_csv(
        outdir / "comparable_pairs.csv",
        index=False,
    )

    non_comparable_df.to_csv(
        outdir / "non_comparable_pairs.csv",
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = {

        "total_pairs": int(
            len(df_comp)
        ),

        "comparable_pairs": int(
            (
                df_comp["STAGE1_SCORE"] == 1
            ).sum()
        ),

        "non_comparable_pairs": int(
            (
                df_comp["STAGE1_SCORE"] == 0
            ).sum()
        ),

        "comparable_ratio": float(
            (
                df_comp["STAGE1_SCORE"] == 1
            ).mean()
        ),

        "agreement_mean": float(
            df_comp["FINAL_SCORE"].mean()
        ),

        "agreement_std": float(
            df_comp["FINAL_SCORE"].std()
        ),
    }

    with open(
        outdir / "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # =====================================================
    # AGGREGATION
    # =====================================================

    agg_df = df_comp.groupby(
        [
            "VIDEO",
            "QUESTION_NUM",
            "BLOCK",
            "VIDEO_SECTOR",
            "AGENT_I",
            "AGENT_J",
        ],
        as_index=False,
    )["FINAL_SCORE"].mean()

    agg_df2 = agg_df.groupby(
        [
            "AGENT_I",
            "AGENT_J",
            "BLOCK",
            "VIDEO_SECTOR",
        ],
        as_index=False,
    )["FINAL_SCORE"].mean()

    # =====================================================
    # PLOT
    # =====================================================

    out_path = outdir / "judge_scores.png"

    plot_judge(
        agg_df2,
        agents=df_answers["AGENT"].unique(),
        df_answers=df_answers,
        out_path=out_path,
    )

    return out_path