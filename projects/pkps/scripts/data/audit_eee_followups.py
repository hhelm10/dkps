"""Verify reconstructed full targets and preserve snapshot/truncation evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(source, prepared, out):
    runs = json.loads((prepared/'run_metadata.json').read_text())
    target = pd.read_parquet(prepared/'targets.parquet').set_index(['model_id', 'task_id'])
    rows, summary = [], {}
    task_name = {
        'gpqa-diamond': lambda n: 'gpqa_diamond', 'gsm-mc': lambda n: 'gsm_mc',
        'math-mc': lambda n: 'math_mc_'+n.replace('_', ''),
        'judgebench': lambda n: 'judge_judgebench_'+n,
        'reward-bench-2': lambda n: 'judge_rewardbench2_'+n,
    }
    for run in runs:
        raw = json.loads((source/run['path']).read_text())
        for ev in raw['evaluation_results']:
            name = ev['evaluation_name'].lower().replace(' ', '_')
            task = run['suite']+':'+task_name[run['suite']](name)
            computed = float(target.loc[(run['model_id'], task), 'y_full'])
            recorded = float(ev['score_details']['score'])
            rows.append(dict(model_id=run['model_id'], task_id=task, computed=computed,
                             recorded=recorded, difference=computed-recorded))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out/'target_score_verification.csv', index=False)
    if len(comparison) != len(target) or not np.allclose(comparison.computed, comparison.recorded, atol=1e-10, rtol=0):
        raise ValueError('reconstructed scores disagree with companion aggregates; inspect verification CSV')
    target.reset_index().groupby('task_id').agg(
        n_models=('model_id', 'nunique'), min_depth=('depth', 'min'), max_depth=('depth', 'max')
    ).to_csv(out/'task_depths.csv')
    for bench in sorted({r['suite'] for r in runs}):
        subset = [r for r in runs if r['suite'] == bench]
        configs = Counter(json.dumps(e.get('generation_config'), sort_keys=True) for r in subset for e in r['evaluations'])
        sources = Counter(json.dumps(e.get('source_data'), sort_keys=True) for r in subset for e in r['evaluations'])
        summary[bench] = dict(n_runs=len(subset),
            generation_configs=[dict(config=json.loads(k), count=v) for k, v in configs.items()],
            source_definitions=[dict(source=json.loads(k), count=v) for k, v in sources.items()])
    (out/'dataset_metadata_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'{len(comparison)} full-depth target cells verified; max error {comparison.difference.abs().max():.3g}')


def token_lengths(prepared, model, out):
    from transformers import AutoTokenizer
    pool = pd.read_parquet(prepared/'pool.parquet')
    texts = list(dict.fromkeys(pool['query'].tolist()+pool['response'].tolist()))
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    lengths = []
    for start in range(0, len(texts), 128):
        result = tokenizer(['task: sentence similarity | query: '+x for x in texts[start:start+128]],
                           truncation=False, padding=False, return_length=True, return_attention_mask=False)
        lengths.extend(result['length'])
    lengths = np.asarray(lengths)
    limit = json.loads((prepared/'embeddings/config.json').read_text())['max_length']
    stats = dict(n_unique_texts=len(lengths), limit_tokens=limit,
                 n_truncated=int((lengths > limit).sum()), fraction_truncated=float((lengths > limit).mean()),
                 quantiles=dict(zip(['min', 'median', 'p90', 'p95', 'p99', 'max'],
                                     np.quantile(lengths, [0, .5, .9, .95, .99, 1]).tolist())),
                 total_untruncated_tokens=int(lengths.sum()),
                 total_limited_tokens=int(np.minimum(lengths, limit).sum()))
    (out/'token_lengths.json').write_text(json.dumps(stats, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--prepared', type=Path, default=Path('data/eee-followups'))
    ap.add_argument('--out', type=Path, default=Path('projects/pkps/artifacts/stress-test'))
    ap.add_argument('--token-lengths', action='store_true')
    ap.add_argument('--model', type=Path, default=Path('data/models/embeddinggemma-2'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audit(args.source, args.prepared, args.out)
    if args.token_lengths:
        token_lengths(args.prepared, args.model, args.out)


if __name__ == '__main__':
    main()
