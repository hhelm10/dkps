"""Token pruning for judge (Model 2) input: drop known low-information trace
content while preserving agent behavior.

Principle: the judge describes what the AGENT did. Repository content the
agent merely displayed (file views, listings), harness state dumps, and
format echoes are not behavior -- they are the dominant token cost and can
be squashed to stubs. Kept intact: agent prose/reasoning, commands, edits
and diffs, error lines, test-outcome summaries.

Rules (format-agnostic):
  1. dedupe     exact repeated paragraphs (harness format echoes)
  2. file view  runs of line-numbered lines (`123:` / `123|`) -> first 3 +
                last 2 lines + a stub
  3. state dump one-line JSON harness state ({"open_file": ...}) -> dropped
                after first occurrence
  4. listing    long directory/file listings -> first 5 entries + stub
  5. code dump  unnumbered code-like runs > `code_keep` lines that are not
                diffs -> head/tail + stub (viewed file bodies, cat output)
  6. test out   long test/exec output -> first 2 + last 6 lines (verdict
                lives at the end)

Usage:
    from dkps.traces.prune import prune_for_judging
    lean = prune_for_judging(trace_text)
"""
import re

_NUM = re.compile(r'\s*\d+[:|\t]')
_JSONSTATE = re.compile(r'\s*\{.*("open_file"|"working_dir"|"command")')
_LISTING = re.compile(r'\s*[\w./\\-]+\.(py|txt|rst|cfg|toml|yml|yaml|json|md|ini)\s*$')
_TEST = re.compile(r'Traceback|FAILED|PASSED|passed|failed|=====+|ERROR|pytest|error:')
_DIFF = re.compile(r'[+-][^+-]|\@\@|diff --git|index [0-9a-f]')
_CODEISH = re.compile(r'^\s{4,}|^\s*(def |class |import |from |return |if |for '
                      r'|while |try:|except|@|#)|[;{}]\s*$')


def _squash(lines, head, tail, label):
    if len(lines) <= head + tail + 2:
        return lines
    return (lines[:head]
            + [f'[... {len(lines) - head - tail} {label} lines omitted ...]']
            + (lines[-tail:] if tail else []))


def prune_for_judging(text, code_keep=12, seen_state=None):
    out_paras = []
    seen = set()
    state_shown = False
    for para in re.split(r'\n\s*\n', text):
        norm = re.sub(r'\s+', ' ', para).strip()
        if not norm:
            continue
        key = norm[:400]
        if key in seen:                                   # rule 1: echo
            continue
        seen.add(key)
        lines = para.split('\n')
        nl = len(lines)
        frac = lambda rx: sum(bool(rx.match(l)) for l in lines) / nl  # noqa: E731

        if frac(_NUM) > 0.4:                              # rule 2: file view
            lines = _squash(lines, 3, 2, 'file-view')
        elif nl <= 2 and any(_JSONSTATE.match(l) for l in lines):   # rule 3
            if state_shown:
                continue
            state_shown = True
        elif frac(_LISTING) > 0.5 and nl > 6:             # rule 4: listing
            lines = _squash(lines, 5, 0, 'listing')
        elif (sum(bool(_TEST.search(l)) for l in lines) / nl > 0.2
              and nl > 10):                               # rule 6: test out
            lines = _squash(lines, 2, 6, 'test-output')
        elif (nl > code_keep
              and sum(bool(_DIFF.match(l)) for l in lines) / nl < 0.3
              and sum(bool(_CODEISH.match(l)) for l in lines) / nl > 0.5):
            lines = _squash(lines, 4, 3, 'code')          # rule 5: code dump
        out_paras.append('\n'.join(lines))
    return '\n\n'.join(out_paras)
