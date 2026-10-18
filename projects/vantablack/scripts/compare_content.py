"""Compare context-conditioning variants by cutoff-free ranking recall.

    python compare_content.py --csv ridge=data/results/rank_eval.csv pkps=data/results/rank_eval_pkps.csv ...

Each CSV comes from rank_eval.py over the 10 planted runs and the real run.
Ring recall is averaged over the five planted runs at each content shift;
sibling-family recall is read from the real (unplanted) run. Writes
writing/notes/content_comparison.md.
"""
import argparse
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from pipeline.paths import NOTES  # noqa: E402
RANKINGS = ['comove', 'intensity', 'coact', 'reply', 'either_rank', 'any3_rank', 'random']
COLS = ['rec@10gr', 'rec@25gr', 'rec@50gr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True, help='name=path')
    a = ap.parse_args()
    rows = []
    for spec in a.csv:
        name, path = spec.split('=', 1)
        df = pd.read_csv(path)
        df['shift'] = df.run.str.extract(r'_a([0-9.]+)_s')[0]
        for (lab, shift, rk), g in df[df.ranking.isin(RANKINGS)].groupby(
                ['labels', df['shift'].fillna('real'), 'ranking']):
            if (lab == 'ring') == (shift == 'real'):
                continue  # ring: planted runs; siblings: the real run
            rows.append(dict(content=name, target=f'{lab} ({"shift " + shift if lab == "ring" else "real"})',
                             ranking=rk, n_runs=len(g), **g[COLS].mean().round(2).to_dict()))
    out = pd.DataFrame(rows).sort_values(['target', 'ranking', 'content'])
    md = ['# Content conditioning: ridge vs PKPS, ranking recall', '',
          'Share of known coordinated pairs that sit together in the top 10 / 25 / 50 ranked groups '
          '(groups of at most 100 agents). Ring: mean over 5 planted runs per shift. Siblings: '
          'the real run. `coact` does not use content and should not change.', '']
    for tgt, g in out.groupby('target', sort=False):
        md += [f'## {tgt}', '', '| ranking | content | top 10 | top 25 | top 50 |', '|---|---|---|---|---|']
        md += [f'| {r.ranking} | {r.content} | {r["rec@10gr"]:.2f} | {r["rec@25gr"]:.2f} | {r["rec@50gr"]:.2f} |'
               for _, r in g.iterrows()]
        md.append('')
    open(os.path.join(NOTES, 'content_comparison.md'), 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
