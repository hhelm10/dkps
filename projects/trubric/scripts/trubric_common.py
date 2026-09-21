"""Shared paths + artifact-saving for the trubric project scripts.

Every trubric_*.py script imports this first; it pins the working
directory to the repo root (so repo-relative data paths resolve) and
exposes save_artifact / save_text_artifact, which write the canonical
artifact files into projects/trubric/artifacts/ (git history is the
version control).
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    '..', '..', '..'))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

DATA = 'projects/trubric/data'
ART = 'projects/trubric/artifacts'


def save_artifact(fig, name, dpi=200, pad=0.03, facecolor=None):
    """PNG (raster preview) + vector PDF (for the paper)."""
    kw = dict(dpi=dpi, bbox_inches='tight', pad_inches=pad)
    if facecolor is not None:
        kw['facecolor'] = facecolor
    for ext in ('png', 'pdf'):
        fig.savefig(f'{ART}/{name}.{ext}', **kw)
    print(f'wrote {ART}/{name}.png/.pdf')


def save_text_artifact(name, ext, text):
    open(f'{ART}/{name}.{ext}', 'w').write(text)
    print(f'wrote {ART}/{name}.{ext}')
