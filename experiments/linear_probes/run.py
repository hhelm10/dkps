"""Portable command line: python run.py --manifest /path/to/manifest.json --out results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from core import evaluate, normalize_blocks


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_embeddings(path, row_ids):
    """Join explicit IDs; never infer correspondence from array position."""
    with np.load(path, allow_pickle=False) as data:
        ids = data['row_id'].astype(str).tolist()
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate embedding row_id in {path}')
        if set(ids) != set(row_ids):
            raise ValueError(f'Embedding/metadata row_id mismatch in {path}')
        positions = {value: i for i, value in enumerate(ids)}
        x = data['X'][[positions[value] for value in row_ids]]
    return normalize_blocks(x)


def write_tables(result, output):
    fields = ['representation', 'target', 'n_rows', 'n_classes', 'balanced_accuracy',
              'ci_low', 'ci_high', 'chance_balanced_accuracy', 'accuracy', 'majority_accuracy']
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in result['results']:
            row = {key: item[key] for key in fields if key in item}
            row['ci_low'], row['ci_high'] = item['balanced_accuracy_ci95']
            writer.writerow(row)
    lines = ['# Linear-probe results', '',
             'Held-out balanced accuracy (%); brackets are 95% group-bootstrap intervals.',
             'Higher means the attribute is more linearly recoverable, not necessarily better.', '',
             '| Representation | Target | N | Classes | Balanced accuracy [95% CI] | Chance |',
             '|---|---|---:|---:|---:|---:|']
    for r in result['results']:
        lo, hi = r['balanced_accuracy_ci95']
        lines.append(f"| {r['representation']} | {r['target']} | {r['n_rows']} | {r['n_classes']} | "
                     f"{100*r['balanced_accuracy']:.1f} [{100*lo:.1f}, {100*hi:.1f}] | "
                     f"{100*r['chance_balanced_accuracy']:.1f} |")
    lines += ['', 'See README.md for the classifier, splits, preprocessing, and interpretation limits.', '']
    output.with_suffix('.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='Output prefix, without extension')
    ap.add_argument('--representations', nargs='+', help='Optional subset of manifest names')
    ap.add_argument('--targets', nargs='+', help='Optional subset of manifest target names')
    ap.add_argument('--alphas', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2, .1, 1.])
    ap.add_argument('--outer-folds', type=int, default=5)
    ap.add_argument('--inner-folds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--bootstrap-draws', type=int, default=1000)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    metadata_path = root / manifest['metadata']
    with metadata_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['row_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Metadata row_id must be unique')
    reps = manifest['representations']
    targets = manifest['targets']
    if args.representations:
        reps = {name: reps[name] for name in args.representations}
    if args.targets:
        targets = {name: targets[name] for name in args.targets}
    config = dict(alphas=args.alphas, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                  seed=args.seed, bootstrap_draws=args.bootstrap_draws)
    provenance = dict(manifest_sha256=sha256(args.manifest), metadata_sha256=sha256(metadata_path),
                      code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                      embeddings_sha256={name: sha256(root / path) for name, path in reps.items()})
    result = dict(protocol='linear-ridge-probes-v1', config=config, provenance=provenance,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  target_definitions=targets, results=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.out.with_suffix('.json')
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        if any(previous[key] != result[key] for key in ('protocol', 'config', 'provenance', 'target_definitions')):
            raise ValueError('Existing output has a different configuration; choose a new output prefix')
        result = previous
    completed = {(r['representation'], r['target']) for r in result['results']}
    for name, path in reps.items():
        if all((name, target) in completed for target in targets):
            continue
        x = load_embeddings(root / path, ids)
        kernel = x @ x.T
        dimensions = x.shape[1]
        del x
        for target, spec in targets.items():
            if (name, target) in completed:
                continue
            start = time.monotonic()
            keep = np.array([bool(r[spec['label']]) and bool(r[spec['group']]) for r in rows])
            selected = np.flatnonzero(keep)
            labels = np.array([rows[i][spec['label']] for i in selected])
            groups = np.array([rows[i][spec['group']] for i in selected])
            print(f'{name} / {target}: {len(selected)} rows, {len(np.unique(labels))} classes', flush=True)
            scores = evaluate(kernel[np.ix_(selected, selected)], labels, groups, **config)
            scores.update(representation=name, target=target, dimensions=dimensions,
                          row_ids=[ids[i] for i in selected], seconds=time.monotonic() - start)
            result['results'].append(scores)
            output_path.write_text(json.dumps(result, indent=2) + '\n')
            write_tables(result, args.out)
            print(f"  balanced accuracy {scores['balanced_accuracy']:.4f}; {scores['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
