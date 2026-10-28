#!/usr/bin/env python
"""
    joint.extract
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data.parsers import parsers

# --
# Parse args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',    type=str, default='./crfm-helm-public/lite/benchmark_output/runs/')
    parser.add_argument('--dataset', type=str, default='math')
    parser.add_argument('--outdir',  type=str, default='data')
    args = parser.parse_args()
    
    args.root   = Path(args.root)
    args.outdir = Path(args.outdir)
    
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    return args

args = parse_args()

# --
# Run

inpaths = sorted(args.root.rglob(f'*/{args.dataset}*/scenario_state.json'))

all_dfs = []
for inpath in tqdm(inpaths, desc='loading data'):
    run = json.loads(inpath.read_text())    
    dps = json.loads((inpath.parent / 'display_predictions.json').read_text())
    
    def dir2dataset(dir):
        if args.dataset == 'med_qa':
            return 'med_qa'
        else:
            return dir.split(',')[0]
    
    _params = {    
        "dataset" : dir2dataset(inpath.parent.name),
        **dict([x.split('=') for x in inpath.parent.name.split(':')[1].split(',')])
    }

    all_dfs.append(parsers[args.dataset].parse(run, dps, _params))


df = pd.concat(all_dfs).reset_index(drop=True)
df = parsers[args.dataset].postprocess(df)

# --
# QC

# sometimes a record is missing some annotations - let's drop them for now
assert all(df.score.notna()), 'some instances are missing scores'
assert df.drop_duplicates(['dataset', 'model', 'instance_id']).shape[0] == df.shape[0]

# --
# Save

df.to_csv(args.outdir / f'{args.dataset}.tsv', index=False, sep='\t')

with open(args.outdir / f'{args.dataset}-response.jl', 'w') as f:
    for _, row in df.iterrows():
        _ = f.write(json.dumps({
            'text': row.response, # TODO: add query and target
        }) + '\n')