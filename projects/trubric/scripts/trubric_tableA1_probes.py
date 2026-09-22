"""Appendix table A1: balanced accuracies underlying Figure 1 (radar).

Rows: text encoder x representation (raw / generic / trubric); columns:
the five trace attributes of Section 3.1. Reads the probe-runner outputs
projects/trubric/data/radar_probes{,_gen}.json and writes
projects/trubric/artifacts/trubric_tableA1_probes.tex.
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from trubric_common import DATA, save_text_artifact  # noqa

ENC = [('openai', 'text-embedding-3-small'), ('nomic', 'nomic-embed-text-v1.5'),
       ('bge', 'bge-large-en-v1.5'), ('gte', 'gte-large'),
       ('e5', 'e5-large-v2'), ('mpnet', 'all-mpnet-base-v2'),
       ('minilm', 'all-MiniLM-L6-v2')]
REPS = [('raw', 'raw trace'), ('generic', 'generic'), ('qubric', r'\trubric{}')]
ATTRS = [('task', '(i) task'), ('outcome', '(ii) correctness'),
         ('system', '(iii) system'), ('vendor', '(iv) family'),
         ('harness', '(v) harness')]


def main():
    table = {}
    chance = {}
    for f in (f'{DATA}/radar_probes.json', f'{DATA}/radar_probes_gen.json'):
        for r in json.load(open(f))['results']:
            table[(r['representation'], r['target'])] = r['balanced_accuracy']
            chance[r['target']] = r['chance_balanced_accuracy']

    out = [r'\begin{table}[t]', r'\centering',
           r'\caption{Held-out balanced accuracy of linear probes for the '
           r'five trace attributes of Section~\ref{sec:method-trubric}, for '
           r'each text encoder and trace representation; these values '
           r'underlie Figure~\ref{fig:radar}. For evaluation-relevant '
           r'attributes (i)--(ii), higher is better; for system-metadata '
           r'attributes (iii)--(v), lower is better. The final row reports '
           r'chance balanced accuracy.}',
           r'\label{tab:probes}',
           r'\small',
           r'\begin{tabular}{llccccc}', r'\toprule',
           r'encoder & representation & ' +
           ' & '.join('{' + lab + '}' for _, lab in ATTRS) + r' \\',
           r'\midrule']
    for short, enc in ENC:
        for k, (rep, replab) in enumerate(REPS):
            key = f'{short}_{rep}'
            cells = ' & '.join(f'{table[(key, a)]:.2f}' for a, _ in ATTRS)
            lead = r'\texttt{' + enc.replace('_', r'\_') + '}' if k == 0 else ''
            out.append(f'{lead} & {replab} & {cells} \\\\')
        out.append(r'\addlinespace[2pt]')
    out += [r'\midrule',
            r'\multicolumn{2}{l}{chance} & ' +
            ' & '.join(f'{chance[a]:.2f}' for a, _ in ATTRS) + r' \\',
            r'\bottomrule', r'\end{tabular}', r'\end{table}']
    save_text_artifact('trubric_tableA1_probes', 'tex', '\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
