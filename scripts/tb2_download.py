"""Terminal-Bench 2.0 leaderboard data download (per HH 2026-09-09).

Source: hf.co/datasets/harborframework/terminal-bench-2-leaderboard
(121.6 GB total; 76 submissions x ~5 jobs x 89 tasks). We pull only what
the analysis needs:

  outcomes   every result.json + metadata.yaml (~100 MB) -> B matrix, y,
             LLM/harness tags
  traces     agent text for ONE job per submission (stdout/command/
             trajectory files; excludes verifier output, recordings,
             setup logs) -- deferred to the 'traces' stage

Usage: python scripts/tb2_download.py outcomes|traces
"""
import sys

from huggingface_hub import snapshot_download

REPO = 'harborframework/terminal-bench-2-leaderboard'
DEST = 'data/terminal_bench/tb2'

PATTERNS = {
    'outcomes': ['**/result.json', '**/metadata.yaml', '**/config.json'],
    # trace text only; recordings/casts and verifier output excluded
    'traces': ['**/agent/**/*.txt', '**/agent/*.txt', '**/agent/*.json'],
}


def main(stage):
    snapshot_download(
        repo_id=REPO, repo_type='dataset', local_dir=DEST,
        allow_patterns=PATTERNS[stage], max_workers=16)
    print('done:', stage)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outcomes')
