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
import yaml
from tqdm import tqdm

from pipeline.config import PipelineConfig
from pipeline.style import DIVERGING_CMAP, apply_style, save_figure
from pipeline.utils.checkpoint import load_dataframe, save_dataframe
from pipeline.utils.io import load_csv
from pipeline.utils.metrics import get_ordered_agents, get_video_sector
# =========================================================
# QUESTIONS MAP
# =========================================================

def load_questions_map(path: Path) -> dict[tuple[str, int], str]:

    if not path.exists():
        raise FileNotFoundError(
            f"Questions YAML not found: {path}"
        )

    raw = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    ) or {}

    mapping: dict[tuple[str, int], str] = {}
    for video, questions in raw.items():
        if not isinstance(questions, dict):
            continue
        for qkey, meta in questions.items():
            m = re.match(r"Q(\d+)$", str(qkey).strip())
            if not m:
                continue
            qnum = int(m.group(1))
            if isinstance(meta, dict):
                question = str(meta.get("question", "")).strip()
            else:
                question = ""
            mapping[(str(video), qnum)] = question

    return mapping



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
    2. They describe the same object's state (e.g., if the question asks "What is the car doing?", any description of its motion—turning, stopping, or going straight—is COMPARABLE because these are competing descriptions of the same event).
    3. The responses are "on the same page" even if they disagree (e.g., "Yes" vs "No").

- Mark as 0 (NOT_COMPARABLE) if:
    1. The responses "talk past each other" (e.g., Question: "What is the car doing?"; A: "It's turning"; B: "It's a blue car"). One describes an action, the other describes an appearance. These are NOT comparable.
    2. One response provides facts while the other says "I don't know," "I can't see," or is empty.
    3. They discuss different objects entirely.

[Few-Shot Examples]
Question: "What is the ego vehicle's action?"
A: "Turning right." | B: "Moving forward." -> 1 (Comparable: These are two different descriptions of the vehicle's trajectory. If one is true, the other is likely false.)
A: "Braking." | B: "Stopped." -> 1 (Comparable: These both describe the vehicle's speed/state.)
A: "Accelerating." | B: "The car is black." -> 0 (Not Comparable: A describes motion, B describes color. They do not overlap or conflict.)

Question: "Is there a traffic light?"
A: "Yes, it is green." | B: "No." -> 1 (Comparable: One confirms existence, the other denies it.)

Output ONLY valid JSON:
{
  "Evaluation": "Briefly explain your reasoning.",
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

Rules:
1. **Conclusion Priority:** If both responses reach the same core conclusion (e.g., both say "Yes," both say "Safe," or both identify the same action like "Braking"), you MUST score +2.
2. **The "Zoom" Rule (Specificity):** Do not penalize for detail. "A vehicle" and "A red Toyota" are a perfect match (+2) because they describe the same entity without contradiction.
3. **The "Bonus Fact" Rule (+1):** Use +1 ONLY if the responses agree on the core answer, but one response includes an *additional, separate factual claim* that the other does not mention (e.g., A: "The light is red"; B: "The light is red and there is a pedestrian").
4. **Contradictions:** Use negative scores if the responses make claims that cannot both be true.

Scoring Scale:
+2 (Strong Agreement): Same core conclusion. This includes cases where one is simply more specific/descriptive than the other (e.g., "moving" vs "accelerating").
+1 (Partial Agreement): Agreement on the core fact, but one response mentions an additional, unrelated detail about the scene that the other omits.
-1 (Partial Contradiction): Agreement on the object/action, but a disagreement on the *degree* or *intensity* (e.g., "moving fast" vs "moving slowly").
-2 (Direct Contradiction): Logically opposite claims (e.g., "Turning" vs "Straight", "Red" vs "Green", "Yes" vs "No").

[Few-Shot Examples]

Question: "What is the ego vehicle doing?"
A: "It is moving." | B: "The vehicle is accelerating forward."
-> Score: 2 (Reason: Both agree on the core action of motion. B is just more specific).

Question: "Is there a car in front?"
A: "Yes." | B: "Yes, and it is a blue truck."
-> Score: 2 (Reason: The core conclusion to the question is identical).

Question: "What is the traffic light color?"
A: "Red." | B: "Red. Also, the road is wet."
-> Score: 1 (Reason: They agree on the light, but B adds a separate fact about the weather/road).

Question: "How is the car moving?"
A: "Moving fast." | B: "Moving slowly."
-> Score: -1 (Reason: They agree it is moving, but contradict on the degree of speed).

Question: "What is the ego vehicle's action?"
A: "Turning right." | B: "Moving forward in the middle lane."
-> Score: -2 (Reason: These are mutually exclusive trajectories).

Output ONLY valid JSON:
{
  "Evaluation": "Briefly explain your reasoning.",
  "Score": 2 | 1 | -1 | -2
}
"""


# =========================================================
# USER TEMPLATE
# =========================================================

USER_TEMPLATE = """
Input:
[Question]: {question}
[Response A]: {res_a}
[Response B]: {res_b}
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
        return json.loads(text, strict=False)

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
            match.group(0),
            strict=False,
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
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "top_k": 20,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
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
    max_retries: int = 3,
    base_delay: float = 2.0,
):

    ctx = (
        f"{row.get('AGENT_I')}|{row.get('AGENT_J')}"
        f"|{row.get('VIDEO')}|Q{row.get('QUESTION_NUM')}"
    )

    async with sem:

        last_err = None

        for attempt in range(1, max_retries + 1):

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

                score = result.get("Score", np.nan)
                evaluation = result.get("Evaluation", "")
                reasoning = result_payload.get("reasoning_content", "")

                return score, evaluation, reasoning

            except Exception as e:

                last_err = e

                if attempt < max_retries:

                    delay = base_delay * (2 ** (attempt - 1))

                    print(
                        f"RETRY {attempt}/{max_retries} "
                        f"({type(e).__name__}): {e!r} | {ctx} | wait {delay}s"
                    )

                    await asyncio.sleep(delay)

        print(
            f"GAVE UP ({type(last_err).__name__}): {last_err!r} | {ctx}"
        )

    return np.nan, "", ""


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
    max_retries: int = 3,
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
                        max_retries=max_retries,
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
                        max_retries=max_retries,
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

            save_dataframe(df, checkpoint_path)

    # =====================================================
    # FINAL SAVE
    # =====================================================

    save_dataframe(df, checkpoint_path)

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

    cmap = DIVERGING_CMAP

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

            # Pairs are deduped at construction (PAIR_KEY sorts AGENT_I/J),
            # so the pivot only fills one triangle. Mirror across the
            # diagonal so the heatmap is fully populated.
            rsa_matrix = rsa_matrix.combine_first(
                rsa_matrix.T
            ).reindex(
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

    save_figure(
        fig,
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
    temperature: float = 1.0,
    max_tokens: int = 8192,
    concurrency: int = 16,
    batch_size: int = 128,
    checkpoint_every_batches: int = 10,
    max_retries: int = 3,
    agents: list[str] | None = None,
) -> Path:

    data_path = config.resolve(
        config.data_file
    )

    outdir = config.out_path(
        "judge"
    )

    # Isolate test runs (with --agents filter) from full-run artifacts.
    suffix = "_test" if agents else ""

    checkpoint_path = (
        outdir
        / f"llm_agreement_scores{suffix}.parquet"
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

    if agents:
        df_answers = df_answers[
            df_answers["AGENT"].isin(agents)
        ].reset_index(drop=True)
        print(
            f"Filtered to {len(df_answers)} rows "
            f"across {df_answers['AGENT'].nunique()} agent(s): {agents}"
        )
        if df_answers["AGENT"].nunique() < 2:
            raise ValueError(
                "Need at least 2 distinct agents present in the data to form pairs."
            )

    if "QUESTION" not in df_answers.columns:
        questions_path = (
            config.workspace_root
            / "final_questions_v3.yaml"
        )
        questions_map = load_questions_map(
            questions_path
        )

        def lookup_question(row) -> str:
            try:
                key = (
                    str(row["VIDEO"]),
                    int(row["QUESTION_NUM"]),
                )
            except Exception:
                return ""
            return questions_map.get(key, "")

        df_answers["QUESTION"] = (
            df_answers.apply(
                lookup_question,
                axis=1,
            )
        )

    df_answers["VIDEO_SECTOR"] = (
        df_answers["VIDEO"].apply(
            get_video_sector
        )
    )

    # =====================================================
    # LOAD OR BUILD CHECKPOINT
    # =====================================================

    df_comp = load_dataframe(checkpoint_path)

    if df_comp is not None:

        print(
            f"Loading checkpoint: {checkpoint_path} ({len(df_comp)} rows)"
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
                "QUESTION",
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

    # Create the client and run the scoring inside a single event loop —
    # otherwise client.aclose() runs on a fresh loop and the transport's
    # original loop is already closed, raising "Event loop is closed" and
    # skipping the split / summary / plot below.
    async def _run_scoring() -> pd.DataFrame:
        client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=None,
        )
        try:
            return await score_dataframe_async(
                df_comp,
                str(checkpoint_path),
                client,
                model,
                temperature,
                max_tokens,
                concurrency=concurrency,
                batch_size=batch_size,
                checkpoint_every_batches=checkpoint_every_batches,
                max_retries=max_retries,
            )
        finally:
            await client.aclose()

    df_comp = asyncio.run(_run_scoring())

    # =====================================================
    # SPLIT DATAFRAMES (three categories — pending is separate
    # so failed/unprocessed rows do not pollute non_comparable)
    # =====================================================

    mask_comparable = df_comp["STAGE1_SCORE"] == 1
    mask_non_comparable = df_comp["STAGE1_SCORE"] == 0
    mask_pending = df_comp["STAGE1_SCORE"].isna()

    df_comp["IS_COMPARABLE"] = mask_comparable

    comparable_df = df_comp[mask_comparable].copy()
    non_comparable_df = df_comp[mask_non_comparable].copy()
    pending_df = df_comp[mask_pending].copy()

    print(
        f"Split: comparable={len(comparable_df)} "
        f"non_comparable={len(non_comparable_df)} "
        f"pending/failed={len(pending_df)}"
    )

    # =====================================================
    # SAVE DATA
    # =====================================================

    save_dataframe(comparable_df, outdir / f"comparable_pairs{suffix}.parquet")
    save_dataframe(non_comparable_df, outdir / f"non_comparable_pairs{suffix}.parquet")
    save_dataframe(pending_df, outdir / f"pending_pairs{suffix}.parquet")

    comparable_df.to_csv(
        outdir / f"comparable_pairs{suffix}.csv",
        index=False,
    )

    non_comparable_df.to_csv(
        outdir / f"non_comparable_pairs{suffix}.csv",
        index=False,
    )

    pending_df.to_csv(
        outdir / f"pending_pairs{suffix}.csv",
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
        outdir / f"summary{suffix}.json",
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

    out_path = outdir / f"judge_scores{suffix}.png"

    plot_judge(
        agg_df2,
        agents=get_ordered_agents(df_answers["AGENT"].unique()),
        df_answers=df_answers,
        out_path=out_path,
    )

    return out_path