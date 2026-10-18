#!/bin/bash
#SBATCH --job-name=colsim
#SBATCH --array=0-120          # one task per cell; check: python run_grid.py --list | wc -l
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs parts
python run_grid.py --cell $SLURM_ARRAY_TASK_ID --workers $SLURM_CPUS_PER_TASK --R ${R:-200} --B ${B:-200}
# afterwards, per experiment: python -c "import pandas as pd,glob; pd.concat(map(pd.read_csv, glob.glob('parts/*.csv'))).groupby('exp').apply(lambda g: g.to_csv(f'results_{g.name}.csv', index=False))"
