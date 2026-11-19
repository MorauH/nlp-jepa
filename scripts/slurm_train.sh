#!/bin/bash
#SBATCH --job-name=jepa
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40s:1
#SBATCH --output=%x-%j.log

set -e

# Move to the repository root so `python -m jepa` resolves the package.
cd "$(dirname "$0")/.."
mkdir -p runs/jepa

VENV=${VENV:-.venv}
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# Grow allocator segments instead of fragmenting fixed blocks; avoids
# near-OOM allocator thrashing with the dynamic-shape batches.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python -u -m jepa train "$@"
