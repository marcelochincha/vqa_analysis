
import re
from pathlib import Path
import pandas as pd
import argparse
import unicodedata
import json
import re
import numpy as np
import logging

logging.basicConfig(
    filename="clean_block2.log",
    filemode="w",        # "w" = sobrescribe, "a" = append
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


##
## UTILITIES
## 

def normalize_text(s):
    if s is None:
        return ""
    t = str(s).strip()
    t = unicodedata.normalize('NFKC', t)
    t = re.sub(r"\s+", " ", t)
    return t

def parse_qnum(qraw):
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

def sanitize_region(region: str):
    if not region or region == '':
        return 'unknown'
    cleaned = str(region).strip().lower()

    if "lima" in cleaned:
        return "lima"
    if "new york city" in cleaned:
        return "nyc"
    return 'NULL'

##
## PROCESS HUMANS
##
def process_humans(
    input_csv: Path,
    placeholder_missing: str = 'NO ANSWER ERROR',
):
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
    # Normalize ANSWER
    long['ANSWER'] = long['ANSWER'].astype(str).str.strip()
    missing_mask = long['ANSWER'].apply(lambda x: x == '')
    long['ANSWER'] = long['ANSWER'].apply(normalize_text)
    
    #print who are missing
    print(f"Filling {missing_mask.sum()} missing answers with placeholder '{placeholder_missing}'")
    
    num_missing = missing_mask.sum()
    if num_missing > 0:
        long.loc[missing_mask, 'ANSWER'] = placeholder_missing
        print(f"Filled {num_missing} missing answers with placeholder '{placeholder_missing}'")
    # select required columns and dedupe
    out = long[['AGENT', 'VIDEO', 'QUESTION_NUM', 'ANSWER']]
    return out

## 
## PROCESS VLMS
##
def process_vlms(vlm_dir: Path, expected: int = 20, placeholder: str = 'NO ANSWER ERROR'):
    vlm_dir = Path(vlm_dir)
    
    files = sorted([p for p in vlm_dir.glob('*.json') if p.is_file()])
    if not files:
        logger.error('No JSON files found in %s', vlm_dir)
        return None

    rows = []
    stats = {'agents': 0, 'total_rows': 0, 'truncated': 0, 'padded': 0, 'missing_filled': 0}

    for jf in files:
        agent = jf.stem
        stats['agents'] += 1
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning('Failed to load %s: %s', jf.name, e)
            continue

        if not isinstance(data, list):
            logger.warning('File %s does not contain a top-level array; skipping', jf.name)
            continue

        for obj in data:
            video = normalize_text(obj.get('video') or '')
            qraw = obj.get('question')
            qnum = parse_qnum(qraw)
            responses = obj.get('response') or obj.get('responses') or []
            if not isinstance(responses, (list, tuple)):
                responses = [responses]

            if len(responses) > expected:
                logger.warning('%s: %s Q%s has %d responses, truncating to %d', agent, video, qnum, len(responses), expected)
                responses = responses[-expected:]
                print(f"Truncated responses for {agent} {video} Q{qnum} to last {expected} now got : {len(responses)}")
                stats['truncated'] += 1
            if len(responses) < expected:
                logger.error('%s: %s Q%s has only %d responses, padding to %d with placeholder', agent, video, qnum, len(responses), expected)
                raise RuntimeError(f"{agent} {video} Q{qnum} has only {len(responses)} responses, expected {expected}")
            for idx, resp in enumerate(responses, start=1):
                ans = normalize_text(resp)
                if ans == '':
                    ans = placeholder
                    stats['missing_filled'] += 1
                rows.append({'AGENT': agent, 'VIDEO': video, 'QUESTION_NUM': qnum, 'ANSWER': ans})
                stats['total_rows'] += 1

    df = pd.DataFrame(rows, columns=['AGENT', 'VIDEO', 'QUESTION_NUM', 'ANSWER'])
    #out_csv.parent.mkdir(parents=True, exist_ok=True)
    #df.to_csv(out_csv, index=False, encoding='utf-8')
    #logger.info('Wrote %d rows for %d agents to %s', len(df), stats['agents'])
    return df


def main():
    parser = argparse.ArgumentParser(description='Join two CSV files')
    parser.add_argument('--csv_humans', default='data/raw/humans/answers_raw_human.csv', help='Path to the first CSV file')
    parser.add_argument('--vlm_dir', default='data/raw/vlms', help='Path to the second CSV file')
    parser.add_argument('--output', default='data/r2_cleaned.csv', help='Path to the output CSV file')
    args = parser.parse_args()

    df1 = process_humans(args.csv_humans)
    df2 = process_vlms(args.vlm_dir)
    result = pd.concat([df1, df2], ignore_index=True)

    # build DF
    result['REPETITION'] = result.groupby(['AGENT', 'VIDEO', 'QUESTION_NUM']).cumcount() + 1
    result['BLOCK'] = result['QUESTION_NUM'].apply(lambda x: (x - 1) // 5 + 1) # Assuming 5 questions per block
    result = result[['AGENT', 'VIDEO','BLOCK','QUESTION_NUM', 'REPETITION', 'ANSWER']] 

    # CLEAN BLOCK 2 VALUES
    #print all unique values in the BLOCK 2
    def extract_number_with_log(text):
        if not isinstance(text, str):
            return np.nan
        #check if the text has X out of Y pattern and extract the X
        #else do the same as before
        math_pattern = re.search(r'(\d+)\s*out\s*of\s*(\d+)', text, re.IGNORECASE)
        numbers = re.findall(r'\b\d+\b', text)
        
        if math_pattern:
            x = int(math_pattern.group(1))
            y = int(math_pattern.group(2))
            if 1 <= x <= 10 and y >= x:
                new_value = x
                logger.info(f"EXTRACTED: '{text}' -> {new_value} (from pattern '{x} out of {y}')")
                return str(new_value)
        if numbers:
            valid = [int(n) for n in numbers if 1 <= int(n) <= 10]
            new_value = valid[-1] if valid else np.nan
            logger.info(f"EXTRACTED: '{text}' -> {new_value} (from numbers {valid})")
            return str(new_value)
        
        return "nan"

    #normalize ndke
    mask = result["BLOCK"] == 2
    result.loc[mask, "ANSWER"] = result.loc[mask, "ANSWER"].apply(extract_number_with_log)

    # Save to a new CSV file
    result.to_csv(args.output, index=False)
    print("CSV files joined successfully!")

# Read the two CSV files
if __name__ == "__main__":
    main()