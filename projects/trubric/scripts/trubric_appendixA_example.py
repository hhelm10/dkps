"""Appendix A: worked example -- a SWE-bench Verified task, its
instantiated \\trubric{}, and extractions for two contrasting systems.

Instance: django__django-11099 (username validators accept a trailing
newline). Systems: the original SWE-agent + GPT-4 submission (2024-04)
and a recent Claude-Opus-4.5-based agent (2025-12), chosen to
illustrate that the summaries describe execution behavior without
naming the harness or model.

Reads data/judge/q100_rubrics/ + data/judge/q100-qspec-flash0731/ and
the cached problem statement. Writes
projects/trubric/artifacts/trubric_appendixA_example.tex.
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from trubric_common import ART, save_text_artifact  # noqa

Q = 'django__django-11099'
SYSTEMS = [('20240402_sweagent_gpt4', 'System A'),
           ('20251205_sonar-foundation-agent_claude-opus-4-5', 'System B')]
FIELD_ORDER = ('understanding', 'localization', 'reproduction',
               'editing', 'verification', 'final_state')

_ESC = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$',
        '&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def esc(t):
    return ''.join(_ESC.get(c, c) for c in t)


def main():
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    stmt = next(r for r in ds if r['instance_id'] == Q)['problem_statement']
    rubric = json.load(open(f'data/judge/q100_rubrics/{Q}.json'))
    extractions = []
    for sysname, label in SYSTEMS:
        j = json.load(open(f'data/judge/q100-qspec-flash0731/{sysname}/{Q}.json'))
        if isinstance(j, list):
            j = j[0]
        extractions.append((label, j))

    out = [r'\section{Example task, \trubric{}, and extractions}',
           r'\label{app:example}',
           '',
           r'We illustrate the \trubric{} pipeline on SWE-bench Verified '
           r'instance \texttt{' + esc(Q) + r'}.',
           '',
           r'\paragraph{Task.} The problem statement, as provided to the '
           r'rubric-instantiation LLM:',
           r'\begin{quote}\small',
           esc(stmt.strip()).replace('\n', r' \\ '),
           r'\end{quote}',
           '',
           r'\paragraph{Instantiated \trubric{}.} The six task-specific '
           r'field descriptions generated for this instance:',
           r'\begin{description}[itemsep=1pt, topsep=2pt]\small']
    for k in FIELD_ORDER:
        out.append(r'\item[' + esc(k.replace('_', ' ')) + r'] '
                   + esc(rubric[k]))
    out.append(r'\end{description}')
    out.append('')
    out.append(r'\paragraph{Extractions.} Structured summaries of two '
               r"systems' traces on this task, produced against the rubric "
               r'above. The summaries describe execution behavior without '
               r'reference to the harness or underlying model that '
               r'produced the trace.')
    for label, ext in extractions:
        out.append('')
        out.append(r'\subparagraph{' + label + r'.}')
        out.append(r'\begin{description}[itemsep=1pt, topsep=2pt]\small')
        for k in FIELD_ORDER:
            out.append(r'\item[' + esc(k.replace('_', ' ')) + r'] '
                       + esc(str(ext.get(k, ''))))
        out.append(r'\end{description}')
    save_text_artifact('trubric_appendixA_example', 'tex',
                       '\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
