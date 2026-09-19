"""Shared paths + artifact-saving for the trubric project scripts.

Every trubric_*.py script imports this first; it pins the working
directory to the repo root (so repo-relative data paths resolve) and
exposes save_artifact / save_text_artifact, which write into
project/trubric/artifacts/ as <name>_<counter>.<ext>. The counter is
content-aware: an unchanged re-render overwrites the latest counter, a
changed artifact gets counter+1, so the directory records iterations
without duplicating identical files.
"""
import glob
import io
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    '..', '..', '..'))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

DATA = 'project/trubric/data'
ART = 'project/trubric/artifacts'


def _next_counter(name, ext, payload):
    paths = glob.glob(f'{ART}/{name}_*.{ext}')
    ks = sorted(int(m.group(1)) for p in paths
                if (m := re.search(rf'_(\d+)\.{ext}$', p)))
    if not ks:
        return 1
    last = f'{ART}/{name}_{ks[-1]}.{ext}'
    mode = 'rb' if isinstance(payload, bytes) else 'r'
    with open(last, mode) as f:
        if f.read() == payload:
            return ks[-1]
    return ks[-1] + 1


def save_artifact(fig, name, dpi=200, pad=0.03, facecolor=None):
    """PNG + vector PDF under one shared counter (PNG bytes decide)."""
    kw = dict(dpi=dpi, bbox_inches='tight', pad_inches=pad)
    if facecolor is not None:
        kw['facecolor'] = facecolor
    buf = io.BytesIO()
    fig.savefig(buf, format='png', **kw)
    data = buf.getvalue()
    k = _next_counter(name, 'png', data)
    open(f'{ART}/{name}_{k}.png', 'wb').write(data)
    fig.savefig(f'{ART}/{name}_{k}.pdf', **kw)
    print(f'wrote {ART}/{name}_{k}.png/.pdf')


def save_text_artifact(name, ext, text):
    k = _next_counter(name, ext, text)
    open(f'{ART}/{name}_{k}.{ext}', 'w').write(text)
    print(f'wrote {ART}/{name}_{k}.{ext}')
