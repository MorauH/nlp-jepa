#!/bin/bash
#SBATCH --job-name=nlp-main-1
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40s:1
#SBATCH --output=%x-%j.log

set -e

cd /data/users/$USER/nlp-jepa

# Grow allocator segments instead of fragmenting fixed blocks; avoids
# near-OOM allocator thrashing with the dynamic-shape batches.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

SEED=1
RUN_DIR="runs/main-seed$SEED"
DEC_DIR="$RUN_DIR-decoder"

singularity exec --nv ./jepa.sif python -u -m jepa train --seed "$SEED" --output-dir "$RUN_DIR"
singularity exec --nv ./jepa.sif python -u -m jepa train --stage decoder --from "$RUN_DIR" --seed "$SEED" --output-dir "$DEC_DIR"
singularity exec --nv ./jepa.sif python -u -m jepa measure --model "$DEC_DIR" --json "$DEC_DIR/metrics.json"
