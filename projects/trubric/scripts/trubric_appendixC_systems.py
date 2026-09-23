"""Appendix C: the evaluated systems and task panels, generated from the
panel data. Writes projects/trubric/artifacts/trubric_appendixC_systems.tex.
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from trubric_common import save_text_artifact  # noqa


def esc(t):
    return t.replace('\\', '').replace('_', r'\_').replace('&', r'\&') \
            .replace('%', r'\%').replace('#', r'\#')


def main():
    import numpy as np
    _sys.path.insert(0, 'scripts')
    from pillars import vendor_tag

    labels = json.load(open('data/leaderboard/verified_labels.json'))
    jd = 'data/judge/structured-qspec'
    swe = sorted(s for s in _os.listdir(jd) if 'resolved' in labels.get(s, {}))
    swe_rows = [(s, vendor_tag(labels, s) or '--',
                 len(labels[s]['resolved']) / 500) for s in swe]

    d = json.load(open('data/terminal_bench/tb2_panel.json'))
    DROP = {'Droid__GPT-5.3-Codex', 'Droid__Claude-Opus-4.6',
            'just-another-coding-agent__GLM-5'}
    all_sys = [s for s in d['systems'] if s not in DROP]
    tasks = d['tasks']
    cov = {s: sum(_os.path.exists(f'data/terminal_bench/tb2_txt/{s}/{t}.txt')
                  for t in tasks) / len(tasks) for s in all_sys}
    kept = [s for s in all_sys if cov[s] >= 0.9]
    meta = d['meta']
    ymap = dict(zip(d['systems'], d['y']))
    tb2_rows = [(s, meta[s]['harness'] or '--',
                 '/'.join(meta[s]['llm_tags']) or '--', ymap[s])
                for s in sorted(kept)]

    q100 = sorted(json.load(open('data/judge/q100.json'))['instances'])

    out = [r'\section{Systems and tasks}', r'\label{app:data}', '',
           r'\subsection{SWE-bench Verified systems}',
           r'Table~\ref{tab:swe-systems} lists the 107 evaluated '
           r'leaderboard submissions with their model-family tag (used by '
           r'the leave-one-family-out protocol; \emph{--} indicates no '
           r'tag could be derived from submission metadata) and official '
           r'resolve rate $y$.', '',
           r'\begin{center}\scriptsize',
           r'\begin{longtable}{llc}',
           r'\caption{SWE-bench Verified systems.}\label{tab:swe-systems}\\',
           r'\toprule', r'system & family & $y$ \\', r'\midrule',
           r'\endfirsthead',
           r'\toprule', r'system & family & $y$ \\', r'\midrule',
           r'\endhead', r'\bottomrule', r'\endlastfoot']
    for s, fam, y in swe_rows:
        out.append(rf'\texttt{{{esc(s)}}} & {esc(fam)} & {y:.3f} \\')
    out += [r'\end{longtable}', r'\end{center}', '',
            r'\subsection{Terminal-Bench 2.0 systems}',
            r'Table~\ref{tab:tb2-systems} lists the 65 evaluated '
            r'submissions with their harness, underlying-model tags, and '
            r'official score $y$ (mean over five trials).', '',
            r'\begin{center}\tiny',
            r'\begin{longtable}{lllc}',
            r'\caption{Terminal-Bench 2.0 systems.}'
            r'\label{tab:tb2-systems}\\',
            r'\toprule', r'system & harness & model tags & $y$ \\',
            r'\midrule', r'\endfirsthead',
            r'\toprule', r'system & harness & model tags & $y$ \\',
            r'\midrule', r'\endhead', r'\bottomrule', r'\endlastfoot']
    for s, h, t, y in tb2_rows:
        out.append(rf'\texttt{{{esc(s)}}} & {esc(h)} & {esc(t)} & {y:.3f} \\')
    out += [r'\end{longtable}', r'\end{center}', '',
            r'\subsection{Task panels}',
            r'The SWE-bench Verified panel consists of the following '
            r'$M = 100$ instances (a fixed 20-task panel united with a '
            r'seeded random draw of 80 further instances):', '',
            r'{\scriptsize\texttt{' + ', '.join(esc(q) for q in q100)
            + r'}}', '',
            r'The Terminal-Bench 2.0 panel uses all 89 tasks:', '',
            r'{\scriptsize\texttt{' + ', '.join(esc(t) for t in sorted(tasks))
            + r'}}']
    save_text_artifact('trubric_appendixC_systems', 'tex',
                       '\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
