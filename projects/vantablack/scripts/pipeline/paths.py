"""Project paths (projects/vantablack/{data, scripts, artifacts, writing}).

data/raw       read-only source archives (gitignored)
data/derived   scopes, embeddings, run pickles and logs (gitignored)
data/results   small result JSON/CSV files the artifacts render from
artifacts      generated figures and tables
writing        paper sources and notes (writing/notes)
"""
import os

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REPO = os.path.abspath(os.path.join(PROJ, '..', '..'))
RAW = os.path.join(PROJ, 'data', 'raw')
DERIVED = os.path.join(PROJ, 'data', 'derived')
RESULTS = os.path.join(PROJ, 'data', 'results')
ART = os.path.join(PROJ, 'artifacts')
NOTES = os.path.join(PROJ, 'writing', 'notes')
