"""Appendix A.3: the verbatim prompts of the trubric pipeline, extracted
directly from scripts/judge_structured.py so the paper cannot drift from
the implementation. Writes
projects/trubric/artifacts/trubric_appendixA_prompts.tex.
"""
import os as _os
import re
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from trubric_common import save_text_artifact  # noqa


def grab(src, name):
    m = re.search(name + r' = """(.*?)"""', src, re.S)
    assert m, name
    return m.group(1).strip('\n')


def main():
    src = open('scripts/judge_structured.py').read()
    qspec = grab(src, 'QSPEC_PROMPT')
    extract = grab(src, 'EXTRACT_PROMPT')
    fixed = grab(src, 'FIXED_RUBRIC')
    out = [r'\subsection{Prompts}', r'\label{app:prompts}', '',
           r'The rubric-instantiation prompt (\texttt{\{problem\}} is the '
           r'task description):',
           r'\begin{quote}\small\begin{verbatim}', qspec,
           r'\end{verbatim}\end{quote}',
           '',
           r'The extraction prompt (\texttt{\{rubric\}} is the instantiated '
           r'\trubric{} for the task, or the fixed generic rubric below for '
           r'the generic baseline; \texttt{\{keys\}} the field names; the '
           r'rendered trace is appended after the prompt):',
           r'\begin{quote}\small\begin{verbatim}', extract,
           r'\end{verbatim}\end{quote}',
           '',
           r'The generic rubric:',
           r'\begin{quote}\small\begin{verbatim}', fixed,
           r'\end{verbatim}\end{quote}']
    save_text_artifact('trubric_appendixA_prompts', 'tex',
                       '\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
