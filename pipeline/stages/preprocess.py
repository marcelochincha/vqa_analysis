from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import PipelineConfig


logger = logging.getLogger(__name__)


def _setup_block2_log(log_path: Path) -> logging.FileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


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


def extract_number_with_log(text, ctx: str = "") -> tuple[str, str]:
    """Return (cleaned_value, action) where action is one of:
       'as_is' | 'extracted_xy' | 'extracted_number' | 'nan_no_match' | 'nan_non_string'
    """
    prefix = f"[{ctx}] " if ctx else ""

    if not isinstance(text, str):
        logger.warning(f"{prefix}NAN: non-string input ({type(text).__name__})")
        return "nan", "nan_non_string"

    stripped = text.strip()
    try:
        as_int = int(float(stripped))
        if 1 <= as_int <= 10:
            logger.info(f"{prefix}AS-IS: '{text}' -> {as_int}")
            return str(as_int), "as_is"
    except (ValueError, TypeError):
        pass

    math_pattern = re.search(r"(\d+)\s*out\s*of\s*(\d+)", text, re.IGNORECASE)
    if math_pattern:
        x = int(math_pattern.group(1))
        y = int(math_pattern.group(2))
        if 1 <= x <= 10 and y >= x:
            logger.info(f"{prefix}EXTRACTED-XY: '{text}' -> {x} (matched '{x} out of {y}')")
            return str(x), "extracted_xy"

    numbers = re.findall(r"\b\d+\b", text)
    valid = [int(n) for n in numbers if 1 <= int(n) <= 10]
    if valid:
        chosen = valid[-1]
        logger.info(f"{prefix}EXTRACTED-NUM: '{text}' -> {chosen} (from candidates {valid})")
        return str(chosen), "extracted_number"

    logger.warning(f"{prefix}NAN: no number in [1,10] found in '{text}'")
    return "nan", "nan_no_match"


def run(config: PipelineConfig, human_csv: Path | None = None, vlm_dir: Path | None = None, output_csv: Path | None = None) -> Path:
    root = config.workspace_root
    human_input = human_csv or root / "data/raw/humans/answers_raw_human.csv"
    vlm_input = vlm_dir or root / "data/raw/vlms"
    output_cleaned = output_csv or root / "data/r2_cleaned.csv"
    output_raw = output_cleaned.parent / "r2.csv"
    log_path = output_cleaned.parent / "preprocess.log"
    audit_path = output_cleaned.parent / "preprocess_block2_audit.csv"

    handler = _setup_block2_log(log_path)
    try:
        print(f"Processing humans from {human_input}...")
        df_humans = process_humans(human_input)

        print(f"Processing VLMs from {vlm_input}...")
        df_vlms = process_vlms(vlm_input)

        result = pd.concat([df_humans, df_vlms], ignore_index=True)
        result["REPETITION"] = result.groupby(["AGENT", "VIDEO", "QUESTION_NUM"]).cumcount() + 1
        result["BLOCK"] = result["QUESTION_NUM"].apply(lambda x: (x - 1) // 5 + 1)
        cols = ["AGENT", "VIDEO", "BLOCK", "QUESTION_NUM", "REPETITION", "ANSWER"]
        result = result[cols]

        # RAW snapshot (block 2 answers preserved in original free-text form)
        output_raw.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_raw, index=False)
        print(f"Saved {len(result)} rows (RAW) to {output_raw}")

        # Block-2 numeric extraction with per-row audit
        result_cleaned = result.copy()
        mask_b2 = result_cleaned["BLOCK"] == 2
        b2_rows = result_cleaned.loc[mask_b2]

        logger.info(f"Starting block-2 normalization on {len(b2_rows)} rows")
        audit_records = []
        new_answers = []
        for idx, row in b2_rows.iterrows():
            ctx = f"{row['AGENT']}|{row['VIDEO']}|Q{row['QUESTION_NUM']}|R{row['REPETITION']}"
            cleaned, action = extract_number_with_log(row["ANSWER"], ctx=ctx)
            new_answers.append(cleaned)
            audit_records.append({
                "AGENT": row["AGENT"],
                "VIDEO": row["VIDEO"],
                "QUESTION_NUM": row["QUESTION_NUM"],
                "REPETITION": row["REPETITION"],
                "ANSWER_RAW": row["ANSWER"],
                "ANSWER_CLEAN": cleaned,
                "ACTION": action,
            })
        result_cleaned.loc[mask_b2, "ANSWER"] = new_answers

        # Audit CSV (one row per block-2 answer with before/after/action)
        audit_df = pd.DataFrame(audit_records)
        audit_df.to_csv(audit_path, index=False)

        # Summary stats
        counts = audit_df["ACTION"].value_counts().to_dict()
        summary = " | ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        logger.info(f"Block-2 normalization done. Counts: {summary}")
        print(f"Block-2 normalization summary: {summary}")

        output_cleaned.parent.mkdir(parents=True, exist_ok=True)
        result_cleaned.to_csv(output_cleaned, index=False)
        print(f"Saved {len(result_cleaned)} rows (CLEANED) to {output_cleaned}")
        print(f"Block-2 audit CSV: {audit_path}")
        print(f"Per-row log:       {log_path}")

        return output_cleaned
    finally:
        logger.removeHandler(handler)
        handler.close()