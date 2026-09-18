"""Compare lexical style signals and exact-format controls, plus readable outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from common import read_jsonl
from style import markers


def summarize(rows):
    styled = [r for r in rows if r.get('kind') == 'style']
    words = sum(len(r['response'].split()) for r in styled)
    counts = sum((Counter(markers(r['response'])) for r in styled), Counter())
    checks = {}
    for row in rows:
        if 'expected' in row:
            checks[row['id']] = row['response'].strip() == row['expected']
        elif 'expected_json' in row:
            try:
                checks[row['id']] = json.loads(row['response']) == row['expected_json']
            except ValueError:
                checks[row['id']] = False
    return {'style_words': words, 'markers_per_1000_words': {k: round(v * 1000 / max(1, words), 2) for k, v in counts.items()},
            'exact_controls': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base')
    parser.add_argument('adapter')
    parser.add_argument('--output', default='outputs/comparison.md')
    args = parser.parse_args()
    base, adapter = read_jsonl(args.base), read_jsonl(args.adapter)
    if {r['id']: r['prompt'] for r in base} != {r['id']: r['prompt'] for r in adapter}:
        raise ValueError('Both runs must use the same prompts')
    summary = {'base': summarize(base), 'adapter': summarize(adapter)}
    print(json.dumps(summary, indent=2))
    lines = ['# Before / after', '', 'Lexical counts are descriptive, not a quality score. Read for helpfulness, factual consistency, naturalness, and unwanted topic changes.', '',
             '```json', json.dumps(summary, indent=2), '```', '']
    adapted = {r['id']: r for r in adapter}
    for row in base:
        lines.extend([f"## {row['id']}", '', row['prompt'], '', '**Base**', '', row['response'], '',
                      '**Adapter**', '', adapted[row['id']]['response'], ''])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))
    print(path)


if __name__ == '__main__':
    main()
