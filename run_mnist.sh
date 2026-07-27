#!/bin/bash
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p slurm_logs
uv run docs/examples/mnist.py \
    --lr=0.001 \
    --num_steps=100000 \
    --mode=hard \
    --task=quantile \
    --num_compare=9 \
    --method=softsort 
    --softness=0.1 \
    --results_csv=docs/examples/mnist/results_quantile.csv \
    --curves_csv=docs/examples/mnist/curves_quantile.csv
