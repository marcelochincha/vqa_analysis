

import re
from pathlib import Path
import pandas as pd
import argparse
import unicodedata


def sanitize_region(region: str):
    if not region or region == '':
        return 'unknown'
    cleaned = str(region).strip().lower()

    if "lima" in cleaned:
        return "lima"
    if "new york city" in cleaned:
        return "nyc"
    return 'NULL'


def process_humans(
    input_csv: Path,
    out_csv: Path,
    placeholder_missing: str = 'NO ANSWER ERROR',
):
    
    #make sure out_csv parent dir exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {input_csv}")
    # detect QA columns
    qa_re = re.compile(r'^(R\d+_\d+)-Q(\d+)$')
    qa_cols = [c for c in df.columns if qa_re.match(c)]
    if not qa_cols:
        raise RuntimeError('No QA columns detected with pattern R##_###-Q#')

    meta_cols = [c for c in df.columns if c not in qa_cols]
    # keep original row index for agent numbering
    df = df.reset_index().rename(columns={'index': 'orig_row'})

    long = df.melt(
        id_vars=['orig_row'] + meta_cols,
        value_vars=qa_cols,
        var_name='QA_COL',
        value_name='ANSWER',
    )

    # extract video id and question num
    def parse_qacol(qacol):
        m = qa_re.match(qacol)
        if not m:
            return None, None
        return m.group(1), int(m.group(2))

    parsed = long['QA_COL'].apply(lambda x: pd.Series(parse_qacol(x), index=['VIDEO_RAW', 'QUESTION_NUM']))
    long = pd.concat([long, parsed], axis=1)

    # AGENT construction: human_{COUNTRY}_{N}
    def make_agent(row):
        country = sanitize_region(row.get('COUNTRY', ''))
        n = int(row['orig_row']) + 1
        return f"human_{country}_{n}"

    long['AGENT'] = long.apply(make_agent, axis=1)

    # VIDEO mapping: replace R2_ -> Robusto2_
    long['VIDEO'] = long['VIDEO_RAW'].astype(str).str.replace(r'^R2_', 'Robusto2_', regex=True)

    # load question strings
    def resolve_question(row):
        qn = int(row['QUESTION_NUM']) if pd.notna(row['QUESTION_NUM']) else None
        return f"Q{qn}"

    long['QUESTION'] = long.apply(resolve_question, axis=1)

    # Normalize ANSWER
    long['ANSWER'] = long['ANSWER'].astype(str).str.strip()
    missing_mask = long['ANSWER'].apply(lambda x: x == '')
    long['ANSWER'] = long['ANSWER'].apply(lambda x: unicodedata.normalize('NFKC', x))
    
    #print who are missing
    print(f"Filling {missing_mask.sum()} missing answers with placeholder '{placeholder_missing}'")
    
    num_missing = missing_mask.sum()
    long.loc[missing_mask, 'ANSWER'] = placeholder_missing

    # select required columns and dedupe
    out = long[['AGENT', 'VIDEO', 'QUESTION_NUM', 'QUESTION', 'ANSWER']].drop_duplicates(
        subset=['AGENT', 'VIDEO', 'QUESTION_NUM'], keep='first'
    )

    out.to_csv(out_csv, index=False)
    summary = {
        'rows_input': int(df.shape[0]),
        'qa_columns': len(qa_cols),
        'rows_output': int(out.shape[0]),
        'missing_answers_filled': int(num_missing),
        'unique_agents': int(out['AGENT'].nunique()),
        'unique_videos': int(out['VIDEO'].nunique()),
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/raw/humans/answers_raw_human.csv')
    p.add_argument('--out', default='data/processed/answers_human.csv')
    p.add_argument('--placeholder', default='NO_ANSWER_ERROR')
    args = p.parse_args()

    summary = process_humans(
        Path(args.input), Path(args.out), args.placeholder
    )
    print('Conversion summary:')
    for k, v in summary.items():
        print(f' - {k}: {v}')


if __name__ == '__main__':
    main()

