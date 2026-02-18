import json
import re
import unicodedata
from pathlib import Path
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("pre_process_vlms")


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


def process_vlms(vlm_dir: Path, out_csv: Path, expected: int = 20, placeholder: str = 'NO ANSWER ERROR'):
    vlm_dir = Path(vlm_dir)
    
    #make sure out dir exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    logger.info('Wrote %d rows for %d agents to %s', len(df), stats['agents'], out_csv)
    summary = {
        'agents_processed': stats['agents'],
        'rows_written': int(len(df)),
        'files_truncated': int(stats['truncated']),
        'files_padded': int(stats['padded']),
        'missing_answers_filled': int(stats['missing_filled'])
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vlm_dir', default='data/raw/vlms')
    p.add_argument('--out', default='data/processed/answers_vlms.csv')
    p.add_argument('--expected', type=int, default=20)
    p.add_argument('--placeholder', default='NO ANSWER ERROR')
    args = p.parse_args()

    summary = process_vlms(Path(args.vlm_dir), Path(args.out), expected=args.expected, placeholder=args.placeholder)
    if summary is None:
        logger.error('No output produced')
        return
    print('Conversion summary:')
    for k, v in summary.items():
        print(f' - {k}: {v}')


if __name__ == '__main__':
    main()

