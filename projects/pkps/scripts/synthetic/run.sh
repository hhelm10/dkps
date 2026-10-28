#!/usr/bin/env bash
set -euo pipefail

PRESET="quick"
OUTDIR="./results/${PRESET}"

python -m projects.pkps.scripts.synthetic.run_block1 --experiment s0 --preset "${PRESET}" --output-dir "${OUTDIR}/s0"
python -m projects.pkps.scripts.synthetic.run_block1 --experiment s1 --preset "${PRESET}" --output-dir "${OUTDIR}/s1"
python -m projects.pkps.scripts.synthetic.run_block1 --experiment s2 --preset "${PRESET}" --output-dir "${OUTDIR}/s2"
python -m projects.pkps.scripts.synthetic.run_block1 --experiment s3 --preset "${PRESET}" --output-dir "${OUTDIR}/s3"
