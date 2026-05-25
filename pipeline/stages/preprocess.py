from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import PipelineConfig


logging.basicConfig(
    filename="preprocess.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t)
    return t


def parse_qnum(qraw) -> int:
    if qraw is None:
        return 0
    s = str(qraw).strip()
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    try:
        return int(s)
    except Exception:
        return 0


def sanitize_region(region: str) -> str:
    if not region or region == "":
        return "unknown"
    cleaned = str(region).strip().lower()
    if "lima" in cleaned:
        return "lima"
    if "new york city" in cleaned:
        return "nyc"
    return "NULL"


def process_humans(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    print(f"Loaded {df.shape[0]} rows from {input_csv}")

    qa_re = re.compile(r"^(R\d+_\d+)-Q(\d+)$")
    qa_cols = [c for c in df.columns if qa_re.match(c)]
    if not qa_cols:
        raise RuntimeError("No QA columns detected with pattern R##_###-Q#")

    meta_cols = [c for c in df.columns if c not in qa_cols]
    df = df.reset_index(drop=True)
    df["orig_row"] = df.index

    long = df.melt(
        id_vars=["orig_row"] + meta_cols,
        value_vars=qa_cols,
        var_name="QA_COL",
        value_name="ANSWER",
    )

    def parse_qacol(qacol):
        m = qa_re.match(qacol)
        if not m:
            return None, None
        return m.group(1), int(m.group(2))

    parsed = long["QA_COL"].apply(lambda x: pd.Series(parse_qacol(x), index=["VIDEO_RAW", "QUESTION_NUM"]))
    long = pd.concat([long, parsed], axis=1)

    def make_agent(row):
        country = sanitize_region(row.get("COUNTRY", ""))
        n = int(row["orig_row"]) + 1
        return f"human_{country}_{n}"

    long["AGENT"] = long.apply(make_agent, axis=1)
    long["VIDEO"] = long["VIDEO_RAW"].astype(str).str.replace(r"^R2_", "Robusto2_", regex=True)
    long["ANSWER"] = long["ANSWER"].astype(str).str.strip()
    long["ANSWER"] = long["ANSWER"].apply(normalize_text)

    return long[["AGENT", "VIDEO", "QUESTION_NUM", "ANSWER"]]


def process_vlms(vlm_dir: Path, expected: int = 20, placeholder: str = "NO ANSWER ERROR") -> pd.DataFrame:
    vlm_dir = Path(vlm_dir)
    files = sorted([p for p in vlm_dir.glob("*.json") if p.is_file()])
    if not files:
        logger.error("No JSON files found in %s", vlm_dir)
        return pd.DataFrame(columns=["AGENT", "VIDEO", "QUESTION_NUM", "ANSWER"])

    rows = []
    stats = {"agents": 0, "total_rows": 0, "truncated": 0, "padded": 0, "missing_filled": 0}

    for jf in files:
        agent = jf.stem
        stats["agents"] += 1
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load %s: %s", jf.name, e)
            continue

        if not isinstance(data, list):
            logger.warning("File %s does not contain a top-level array; skipping", jf.name)
            continue

        for obj in data:
            video = normalize_text(obj.get("video") or "")
            qraw = obj.get("question")
            qnum = parse_qnum(qraw)
            responses = obj.get("response") or obj.get("responses") or []
            if not isinstance(responses, (list, tuple)):
                responses = [responses]

            if len(responses) > expected:
                logger.warning("%s: %s Q%s has %d responses, truncating to %d", agent, video, qnum, len(responses), expected)
                responses = responses[-expected:]
                stats["truncated"] += 1
            if len(responses) < expected:
                logger.error("%s: %s Q%s has only %d responses", agent, video, qnum, len(responses))

            for idx, resp in enumerate(responses, start=1):
                ans = normalize_text(resp)
                if ans == "":
                    ans = placeholder
                    stats["missing_filled"] += 1
                rows.append({"AGENT": agent, "VIDEO": video, "QUESTION_NUM": qnum, "ANSWER": ans})
                stats["total_rows"] += 1

    return pd.DataFrame(rows, columns=["AGENT", "VIDEO", "QUESTION_NUM", "ANSWER"])


def extract_number_with_log(text) -> str:
    if not isinstance(text, str):
        return "nan"
    math_pattern = re.search(r"(\d+)\s*out\s*of\s*(\d+)", text, re.IGNORECASE)
    numbers = re.findall(r"\b\d+\b", text)

    if math_pattern:
        x = int(math_pattern.group(1))
        y = int(math_pattern.group(2))
        if 1 <= x <= 10 and y >= x:
            logger.info(f"EXTRACTED: '{text}' -> {x} (from pattern '{x} out of {y}')")
            return str(x)
    if numbers:
        valid = [int(n) for n in numbers if 1 <= int(n) <= 10]
        if valid:
            new_value = valid[-1]
            logger.info(f"EXTRACTED: '{text}' -> {new_value} (from numbers {valid})")
            return str(new_value)

    return "nan"


def run(config: PipelineConfig, human_csv: Path | None = None, vlm_dir: Path | None = None, output_csv: Path | None = None) -> Path:
    root = config.workspace_root
    human_input = human_csv or root / "data/raw/humans/answers_raw_human.csv"
    vlm_input = vlm_dir or root / "data/raw/vlms"
    output_cleaned = output_csv or root / "data/r2_cleaned.csv"
    output_raw = output_cleaned.parent / "r2.csv"

    print(f"Processing humans from {human_input}...")
    df_humans = process_humans(human_input)

    print(f"Processing VLMs from {vlm_input}...")
    df_vlms = process_vlms(vlm_input)

    result = pd.concat([df_humans, df_vlms], ignore_index=True)

    result["REPETITION"] = result.groupby(["AGENT", "VIDEO", "QUESTION_NUM"]).cumcount() + 1
    result["BLOCK"] = result["QUESTION_NUM"].apply(lambda x: (x - 1) // 5 + 1)
    cols = ["AGENT", "VIDEO", "BLOCK", "QUESTION_NUM", "REPETITION", "ANSWER"]
    result = result[cols]

    # Save the RAW snapshot (block 2 answers still in original free-text form)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_raw, index=False)
    print(f"Saved {len(result)} rows (RAW) to {output_raw}")

    # Apply block-2 numeric extraction and save the CLEANED version
    result_cleaned = result.copy()
    mask_b2 = result_cleaned["BLOCK"] == 2
    result_cleaned.loc[mask_b2, "ANSWER"] = result_cleaned.loc[mask_b2, "ANSWER"].apply(extract_number_with_log)

    output_cleaned.parent.mkdir(parents=True, exist_ok=True)
    result_cleaned.to_csv(output_cleaned, index=False)
    print(f"Saved {len(result_cleaned)} rows (CLEANED, block-2 normalized) to {output_cleaned}")

    return output_cleaned